#!/usr/bin/env python3
"""
Rabbit Pie 単4×2 電池グリップモジュール ジェネレーター
======================================================

okikata.org の Rabbit Pie 用背面ケース (case_fix_2026.stl) の背面に、
単4電池2本を直列で搭載できる「電池グリップモジュール」を追加する
3部品を生成します。元のケース STL は無改造のまま使えます
（既存の Φ6 軽量化穴をスナップリベットの固定穴として利用します）。

生成される部品（すべてサポート不要・平置きで高速印刷可能）:
  1. battery_grip.stl  … 電池室＋グリップ本体
  2. battery_lid.stl   … スライド式の電池蓋（ディテント付き）
  3. snap_rivet_x6.stl … プッシュリベット 6本（4本使用＋予備2本）

必要な市販部品:
  - 電池ボックス用電極（単4/単3用の板電極。幅 ~9.5-10mm, 高さ ~11-12mm）
      * マイナス側バネ電極 ×1、プラス側平板電極 ×1（リード線ハンダ付け）
      * ブリッジ電極（2本を直列接続する連結板）×1
        …なければ単体電極2枚を銅線でつないでも OK
  - リード線 2本 (AWG24-26 程度)

使い方:
  pip install numpy trimesh manifold3d shapely
  python3 generate_battery_grip.py
"""

import numpy as np
import trimesh
from trimesh.creation import box, cylinder, extrude_polygon, triangulate_polygon
from trimesh.transformations import rotation_matrix
import shapely.geometry as sg

# ----------------------------------------------------------------------------
# パラメータ（すべて mm）
# ----------------------------------------------------------------------------

# --- 単4電池 ---
CELL_DIA = 10.5          # 単4直径
CELL_LEN = 44.5          # 単4長さ
CELL_CLR = 1.1           # チャンネル幅の直径方向クリアランス
CAVITY_LEN = 48.0        # 電極面間距離（バネ圧縮分を含む）

# --- ケース側の実測値 (case_fix_2026.stl を解析した値) ---
CASE_BACK_Z = -3.97      # ケース背面（外側平面）の Z
CASE_PLATE_T = 1.0       # 背面プレート厚
CASE_HOLE_DIA = 6.0      # 既存の軽量化穴の直径
# リベットで使う4穴の実測中心座標 (x, y)
RIVET_HOLES = [(-15.40, -9.54), (18.20, -9.51),
               (-15.40, -17.94), (18.20, -17.91)]
# 配線をケース内部へ通すのに使う既存穴（電極タブ側ポケットの直上）
WIRE_HOLES = [(-23.77, -9.48), (-23.77, -17.97)]

# --- モジュール構造 ---
FLOOR_T = 2.0            # ケースに密着する床（合わせ面）の厚み
SIDE_WALL = 3.0          # 側壁（蓋レール溝 1.2 が入るため厚め）
END_WALL = 2.0           # 端壁の外皮（電極ポケットの外側）
RIB_W = 1.6              # 中央仕切りリブ
SLOT_T = 1.0             # 電極板の差し込みスロット厚
SLOT_W = 10.8            # 電極板スロット幅（~10mm 電極用）
SLOT_SINK = 1.0          # スロットを床に食い込ませる深さ（背の高い電極対応）
POCKET_T = 2.0           # 電極背面ポケット（バネ背面・ハンダタブ用）
BAY_DEPTH = 11.5         # 床から蓋下面までの深さ
LID_T = 1.4              # 蓋の板厚
LID_CLR = 0.3            # 蓋まわりのクリアランス
GROOVE_D = 1.2           # 蓋レール溝の深さ（側壁へ）
LEDGE = 0.6              # 溝より外面側に残す壁
CORNER_R = 5.0           # 外形コーナー R
CHAMFER = 1.6            # 外面周囲の 45° 面取り（グリップの当たりを柔らかく）

MOD_CX = 0.0             # モジュール中心 X（ケース座標）
MOD_CY = -13.7           # モジュール中心 Y（既存の穴列に合わせ背面下部に配置）

# --- リベット ---
RIVET_SHAFT_D = 5.7
RIVET_BARB_D = 6.9
RIVET_HEAD_D = 9.0
RIVET_HEAD_T = 1.1
RIVET_GRIP = 1.8         # 床残り 0.8 + ケース板 1.0
RIVET_SLOT = 1.5         # 撓み用スリット幅
MOD_HOLE_DIA = 6.7       # モジュール床のリベット通し穴（ゆるめ）
CBORE_DIA = 9.6          # リベット頭の座ぐり
CBORE_DEPTH = 1.2

EPS = 0.05               # ブーリアン用の逃げ

# ----------------------------------------------------------------------------
# 導出値
# ----------------------------------------------------------------------------
CH_W = CELL_DIA + CELL_CLR                    # チャンネル幅 11.6
CH_Y = [MOD_CY + (CH_W + RIB_W) / 2,          # チャンネル1中心 y=-7.1
        MOD_CY - (CH_W + RIB_W) / 2]          # チャンネル2中心 y=-20.3

BAY_X = CAVITY_LEN / 2                        # 電極面 x=±24.0
SLOT_X0, SLOT_X1 = BAY_X, BAY_X + SLOT_T      # スロット 24.0..25.0
POCK_X0, POCK_X1 = SLOT_X1, SLOT_X1 + POCKET_T  # ポケット 25.0..27.0
MOD_HX = POCK_X1 + END_WALL                   # 外形半長 29.0
MOD_HY = CH_W + RIB_W / 2 + SIDE_WALL         # 外形半幅 15.4
WALL_IN_Y = CH_W + RIB_W / 2                  # 側壁内面 |y-MOD_CY| = 12.4

# Z 座標（組付け座標系: ケース背面が z=-3.97, モジュールは -z 側に張り出す）
Z_TOP = CASE_BACK_Z                           # 合わせ面
Z_FLOOR = Z_TOP - FLOOR_T                     # 電池室の床
Z_BAY = Z_FLOOR - BAY_DEPTH                   # 蓋下面（電池室の口）
Z_LID_BOT = Z_BAY - LID_T
Z_GROOVE_TOP = Z_BAY + 0.1                    # 溝上端（クリアランス 0.1）
Z_GROOVE_BOT = Z_LID_BOT - (LID_CLR - 0.1)    # 溝下端
Z_BOT = Z_GROOVE_BOT - LEDGE                  # モジュール外面
MOD_H = Z_TOP - Z_BOT                         # 背面からの張り出し量

GROOVE_STOP_X = MOD_HX - 0.7                  # 溝止まり（x+ 側は貫通しない）
LID_W = 2 * (WALL_IN_Y + GROOVE_D) - LID_CLR  # 蓋の幅
LID_LEN = MOD_HX + GROOVE_STOP_X - LID_CLR    # 蓋の長さ
DETENT_Y = 12.85                              # ディテントの中心オフセット
DETENT_X = -23.0                              # ディテントの x 位置（挿入口より奥）


def B(x0, x1, y0, y1, z0, z1):
    """軸並行ボックス"""
    b = box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


def rounded_rect(hx, hy, r, cx=0.0, cy=0.0):
    """角丸長方形の shapely ポリゴン"""
    return sg.box(cx - hx + r, cy - hy + r, cx + hx - r, cy + hy - r) \
             .buffer(r, quad_segs=12, join_style=1)


def prism(poly, z0, z1):
    m = extrude_polygon(poly, height=z1 - z0)
    m.apply_translation([0, 0, z0])
    return m


def loft(poly_bot, poly_top, z0, z1):
    """同一頂点数の2つのポリゴン間をロフトした watertight ソリッド"""
    b = np.array(poly_bot.exterior.coords[:-1])
    t = np.array(poly_top.exterior.coords[:-1])
    assert len(b) == len(t), (len(b), len(t))
    n = len(b)
    verts = np.vstack([np.c_[b, np.full(n, z0)], np.c_[t, np.full(n, z1)]])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + i], [j, n + j, n + i]]
    # 上下のキャップ
    vb, fb = triangulate_polygon(poly_bot)
    vt, ft = triangulate_polygon(poly_top)
    o1 = len(verts)
    verts = np.vstack([verts, np.c_[vb, np.full(len(vb), z0)]])
    faces += (fb[:, ::-1] + o1).tolist()
    o2 = len(verts)
    verts = np.vstack([verts, np.c_[vt, np.full(len(vt), z1)]])
    faces += (ft + o2).tolist()
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    trimesh.repair.fix_normals(m)
    return m


def union(parts):
    return trimesh.boolean.union(parts, engine='manifold')


def diff(a, parts):
    return trimesh.boolean.difference([a] + parts, engine='manifold')


# ----------------------------------------------------------------------------
# 1) 電池グリップ本体
# ----------------------------------------------------------------------------
def build_grip():
    # --- 外形: 角丸プリズム + 外面周囲 45° 面取り（ロフトで生成）---
    outline = rounded_rect(MOD_HX, MOD_HY, CORNER_R, MOD_CX, MOD_CY)
    outline_s = rounded_rect(MOD_HX - CHAMFER, MOD_HY - CHAMFER,
                             CORNER_R, MOD_CX, MOD_CY)
    body = union([prism(outline, Z_BOT + CHAMFER, Z_TOP),
                  loft(outline_s, outline, Z_BOT, Z_BOT + CHAMFER)])

    cuts = []
    y_in0 = MOD_CY - WALL_IN_Y                # 側壁内面（下側）
    y_in1 = MOD_CY + WALL_IN_Y                # 側壁内面（上側）

    # --- 電池ベイ（2チャンネル、中央リブを残す）---
    for cy in CH_Y:
        cuts.append(B(-BAY_X, BAY_X, cy - CH_W / 2, cy + CH_W / 2,
                      Z_BAY - EPS, Z_FLOOR))

    # --- 電極スロット（床へ SLOT_SINK 食い込ませ、背の高い電極に対応）---
    slot_z0, slot_z1 = Z_BAY - EPS, Z_FLOOR + SLOT_SINK
    for cy in CH_Y:  # x- 側: チャンネルごとの単体電極
        cuts.append(B(-SLOT_X1, -SLOT_X0, cy - SLOT_W / 2, cy + SLOT_W / 2,
                      slot_z0, slot_z1))
    # x+ 側: 2本を直列接続するブリッジ電極用の幅広スロット
    cuts.append(B(SLOT_X0, SLOT_X1, CH_Y[1] - SLOT_W / 2,
                  CH_Y[0] + SLOT_W / 2, slot_z0, slot_z1))

    # --- 電極背面ポケット（バネの背・ハンダタブの逃げ）---
    for cy in CH_Y:
        cuts.append(B(-POCK_X1, -POCK_X0 + EPS, cy - SLOT_W / 2,
                      cy + SLOT_W / 2, Z_BAY - EPS, Z_FLOOR))
    cuts.append(B(POCK_X0 - EPS, POCK_X1, CH_Y[1] - SLOT_W / 2,
                  CH_Y[0] + SLOT_W / 2, Z_BAY - EPS, Z_FLOOR))

    # --- 外側開口: 側壁内面の間・端壁外皮の間を外面からベイ口まで開ける ---
    #     （蓋がモジュール外面に露出する。蓋は側壁レールと端皮リップで保持）
    cuts.append(B(-POCK_X1, POCK_X1, y_in0, y_in1, Z_BOT - EPS, Z_BAY + EPS))

    # --- 蓋レール溝（側壁内面, x- 端は貫通・x+ 端は 0.7 手前で止め）---
    for y0, y1 in [(y_in1, y_in1 + GROOVE_D), (y_in0 - GROOVE_D, y_in0)]:
        cuts.append(B(-MOD_HX - EPS, GROOVE_STOP_X, y0, y1,
                      Z_GROOVE_BOT, Z_GROOVE_TOP))
    # 端皮の蓋通し: x- 側は挿入口（蓋リブが通るよう天井高め）、x+ 側は溝高さ
    cuts.append(B(-MOD_HX - EPS, -POCK_X1 + EPS, y_in0 - EPS,
                  y_in1 + EPS, Z_GROOVE_BOT, Z_BAY + 1.3))
    cuts.append(B(POCK_X1 - EPS, GROOVE_STOP_X, y_in0 - EPS,
                  y_in1 + EPS, Z_GROOVE_BOT, Z_GROOVE_TOP))

    # --- 配線窓: x- ポケットの床を抜き、ケース既存 Φ6 穴の直上に開口 ---
    for hx, hy in WIRE_HOLES:
        cuts.append(B(-26.9, -24.6, hy - 1.6, hy + 1.6,
                      Z_FLOOR - EPS, Z_TOP + EPS))

    # --- リベット穴 + 頭の座ぐり ---
    for hx, hy in RIVET_HOLES:
        c = cylinder(radius=MOD_HOLE_DIA / 2, height=FLOOR_T + 2 * EPS,
                     sections=48)
        c.apply_translation([hx, hy, (Z_FLOOR + Z_TOP) / 2])
        cuts.append(c)
        cb = cylinder(radius=CBORE_DIA / 2, height=CBORE_DEPTH + EPS,
                      sections=48)
        cb.apply_translation([hx, hy, Z_FLOOR + (CBORE_DEPTH - EPS) / 2])
        cuts.append(cb)

    # --- ディテント用ディンプル（溝の上面, 閉位置で蓋のバンプが嵌まる）---
    for sy in (-1, 1):
        d = trimesh.creation.icosphere(subdivisions=2, radius=0.9)
        d.apply_translation([DETENT_X, MOD_CY + sy * DETENT_Y, Z_GROOVE_TOP])
        cuts.append(d)

    # --- 極性マーク（電池室床に 0.5 彫り込み）---
    def plus(cx, cy):
        return [B(cx - 2, cx + 2, cy - 0.6, cy + 0.6,
                  Z_FLOOR - 0.5, Z_FLOOR + EPS),
                B(cx - 0.6, cx + 0.6, cy - 2, cy + 2,
                  Z_FLOOR - 0.5, Z_FLOOR + EPS)]

    def minus(cx, cy):
        return [B(cx - 2, cx + 2, cy - 0.6, cy + 0.6,
                  Z_FLOOR - 0.5, Z_FLOOR + EPS)]

    # ch1(y=-7.1): + が x+（ブリッジ側）/ ch2(y=-20.3): + が x-（タブ側）
    cuts += plus(10, CH_Y[0] + 3.0) + minus(-10, CH_Y[0] + 3.0)
    cuts += minus(10, CH_Y[1] - 3.0) + plus(-10, CH_Y[1] - 3.0)

    grip = diff(body, cuts)

    # 印刷姿勢へ: 合わせ面(Z_TOP)を下にする → X軸まわりに 180° 回転
    grip_print = grip.copy()
    grip_print.apply_transform(rotation_matrix(np.pi, [1, 0, 0],
                                               point=[0, MOD_CY, 0]))
    grip_print.apply_translation([0, 0, -grip_print.bounds[0][2]])
    return grip, grip_print


# ----------------------------------------------------------------------------
# 2) スライド蓋
# ----------------------------------------------------------------------------
def build_lid():
    x0 = -MOD_HX + LID_CLR / 2
    z_out = Z_LID_BOT + LID_CLR / 2           # 外面
    z_in = Z_BAY - LID_CLR / 2                # 内面（電池側）
    lid = prism(rounded_rect(LID_LEN / 2, LID_W / 2, 1.2,
                             x0 + LID_LEN / 2, MOD_CY), z_out, z_in)
    adds, cuts = [], []

    # 電池押さえリブ（チャンネルごとに1本, ガタつき防止）
    for cy in CH_Y:
        adds.append(B(-BAY_X + 4, BAY_X - 4, cy - 0.6, cy + 0.6,
                      z_in - EPS, z_in + 1.0))

    # ディテントバンプ（レール上面, 溝ディンプルに嵌まる。0.45 突出）
    for sy in (-1, 1):
        b = trimesh.creation.icosphere(subdivisions=2, radius=0.85)
        b.apply_translation([DETENT_X + LID_CLR / 2,
                             MOD_CY + sy * DETENT_Y, z_in - 0.45])
        adds.append(b)

    # 開閉用の滑り止め溝（外面側 3本）
    for i in range(3):
        gx = x0 + 4.0 + i * 2.4
        cuts.append(B(gx - 0.5, gx + 0.5, MOD_CY - 8, MOD_CY + 8,
                      z_out - EPS, z_out + 0.5))

    lid = diff(union([lid] + adds), cuts)

    # 印刷姿勢: 外面(z_out 側)が下 → 平行移動のみ
    lid_print = lid.copy()
    lid_print.apply_translation([0, 0, -lid_print.bounds[0][2]])
    return lid, lid_print


# ----------------------------------------------------------------------------
# 3) スナップリベット
# ----------------------------------------------------------------------------
def cone(r0, r1, z0, z1, sections=64):
    """円錐台 (z0 で半径 r0, z1 で半径 r1)"""
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ring0 = np.c_[r0 * np.cos(theta), r0 * np.sin(theta),
                  np.full(sections, z0)]
    ring1 = np.c_[r1 * np.cos(theta), r1 * np.sin(theta),
                  np.full(sections, z1)]
    verts = np.vstack([ring0, ring1, [[0, 0, z0], [0, 0, z1]]])
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces += [[i, j, sections + i], [j, sections + j, sections + i]]
        faces += [[j, i, 2 * sections],
                  [sections + i, sections + j, 2 * sections + 1]]
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    trimesh.repair.fix_normals(m)
    return m


def build_rivet():
    head = cylinder(radius=RIVET_HEAD_D / 2, height=RIVET_HEAD_T, sections=64)
    head.apply_translation([0, 0, RIVET_HEAD_T / 2])
    z0 = RIVET_HEAD_T
    shaft = cylinder(radius=RIVET_SHAFT_D / 2, height=RIVET_GRIP + EPS,
                     sections=64)
    shaft.apply_translation([0, 0, z0 + (RIVET_GRIP + EPS) / 2 - EPS])
    barb = cone(RIVET_SHAFT_D / 2, RIVET_BARB_D / 2,
                z0 + RIVET_GRIP - EPS, z0 + RIVET_GRIP + 0.6)
    tip = cone(RIVET_BARB_D / 2, 2.2, z0 + RIVET_GRIP + 0.6 - EPS,
               z0 + RIVET_GRIP + 2.2)
    rivet = union([head, shaft, barb, tip])
    # 撓みスリット（頭に 0.5 食い込み）
    slit = B(-RIVET_SLOT / 2, RIVET_SLOT / 2, -RIVET_BARB_D, RIVET_BARB_D,
             RIVET_HEAD_T - 0.5, z0 + RIVET_GRIP + 2.3)
    return diff(rivet, [slit])


def rivet_plate(rivet, n=6, pitch=12.0):
    plate = []
    for i in range(n):
        r = rivet.copy()
        r.apply_translation([(i % 3) * pitch, (i // 3) * pitch, 0])
        plate.append(r)
    return union(plate)


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    os.makedirs('stl', exist_ok=True)

    grip_asm, grip_print = build_grip()
    lid_asm, lid_print = build_lid()
    rivet = build_rivet()
    rivets6 = rivet_plate(rivet)

    for name, mesh in [('battery_grip', grip_print),
                       ('battery_lid', lid_print),
                       ('snap_rivet_x6', rivets6)]:
        path = f'stl/{name}.stl'
        mesh.export(path)
        print(f'{path}: watertight={mesh.is_watertight} '
              f'tris={len(mesh.faces)} '
              f'size={np.round(mesh.extents, 2).tolist()}')

    # 組付け確認用（ケース座標系のまま）
    grip_asm.export('stl/_assembly_grip.stl')
    lid_asm.export('stl/_assembly_lid.stl')
    print('module footprint:', 2 * MOD_HX, 'x', 2 * MOD_HY,
          ' protrusion:', round(MOD_H, 2))
