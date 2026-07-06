#!/usr/bin/env python3
"""
Rabbit Pie 単4×2 電池内蔵ケース ジェネレーター（一体型）
========================================================

okikata.org の Rabbit Pie 背面ケース (case_fix_2026.stl) の下端を延長し、
単4電池2本（直列 3V）を内蔵する「電池バー」を融合した一体型ケース STL を
生成します。

- 追加の印刷部品なし（蓋なし・リベットなし）。印刷するのはこのケース1個だけ
- 電池はスナップ保持: チャンネル壁が電池を約220°包み込み、開口を
  電池径より 0.3mm 狭くしてパチンと固定。交換はバネ側へ押してから持ち上げ
- 背面はケースと同一平面 → 従来どおり背面を下にした平置き印刷・サポート不要
- 両端の「電極タワー」に市販の電池ボックス用電極を上から差し込むだけ
- リード線はケース底壁を貫通するトンネルで直接内部へ

必要な市販部品:
  - 電池ボックス用電極（幅 ~9.5-10mm・高さ ~11-12.5mm の一般的な板電極）
      * マイナス側バネ電極 ×1 / プラス側平板電極 ×1（リード線ハンダ付け）
      * 2本連結ブリッジ電極 ×1（無ければ単体電極2枚を銅線で接続）
  - リード線 2本 (AWG24-26)

使い方:
  pip install numpy trimesh manifold3d shapely
  python3 generate_battery_case.py <元ケースSTL>   # 省略時 stl/case_fix_2026_original.stl
"""

import sys
import numpy as np
import trimesh
from trimesh.creation import box, cylinder, extrude_polygon, triangulate_polygon
from trimesh.transformations import rotation_matrix
import shapely.geometry as sg
import shapely.ops

# ----------------------------------------------------------------------------
# パラメータ（すべて mm、ケース座標系）
# ----------------------------------------------------------------------------

# --- 単4電池 ---
CELL_DIA = 10.5
CELL_LEN = 44.5
CELL_FIT = 0.4           # チャンネル径の余裕（Φ10.9）
SNAP_PINCH = 0.3         # スナップ開口の絞り量（開口幅 = CELL_DIA - これ）
CAVITY_LEN = 48.0        # 電極面間距離（バネ圧縮分込み）

# --- ケース実測値 ---
CASE_BACK_Z = -3.97      # 背面（印刷ベッド面）
CASE_RIM_Z = 4.90        # 前面リム
CASE_HX = 29.81          # ケース半幅（X）
CASE_BOT_Y = -34.65      # 底辺外面
WALL_IN_Y = -32.2        # 底壁内面（実測）

# --- 電池バー ---
FLOOR_T = 1.6            # チャンネル床（ベッド側）
TOP_WALL = 2.2           # ケース側チャンネル外の壁
OUT_WALL = 2.2           # 外側チャンネル外の壁
RIB_W = 1.9              # チャンネル間ピッチ余裕（ピッチ = Φ + これ）
OVERLAP = 0.65           # ケース底壁への食い込み（融合用）
SLOT_T = 1.0             # 電極スロット厚
SLOT_W = 10.8            # 電極スロット幅
SLOT_SINK = 0.8          # スロットの床食い込み
POCKET_T = 2.0           # 電極背面ポケット
SLOT_DEPTH = 12.5        # スロット深さ（電極高さ ~12.5mm まで対応）
CORNER_R = 6.0           # バー下側コーナー R
CHAMFER_F = 1.6          # 前面外周エッジの 45° 面取り
CHAMFER_T = 1.2          # タワー上端エッジの 45° 面取り
MOUTH_LEAD = 0.3         # スナップ開口の面取り（入れやすさ）

# --- 配線 ---
TUNNEL_X = (-19.0, -16.0)          # 底壁貫通トンネル（コーナーポストを回避）
TUNNEL_Z = (-2.8, -0.6)            # 背面プレート上面〜基板ハンダ面の間
DUCT_Z = (-2.37, -1.0)             # バー内の配線ダクト高さ（電池室と分離）

EPS = 0.05

# ----------------------------------------------------------------------------
# 導出値
# ----------------------------------------------------------------------------
R_CH = (CELL_DIA + CELL_FIT) / 2                   # チャンネル半径 5.45
AXIS_Z = CASE_BACK_Z + FLOOR_T + R_CH              # 電池軸 z = 3.08
OPEN_HW = (CELL_DIA - SNAP_PINCH) / 2              # スナップ開口半幅 5.1
LIP_Z = AXIS_Z + np.sqrt(R_CH**2 - OPEN_HW**2)     # リップ z = 5.00
BAR_FRONT = LIP_Z + 0.5                            # バー前面 z = 5.50
CH1_Y = CASE_BOT_Y - TOP_WALL - R_CH               # チャンネル1軸 y = -42.30
PITCH = CELL_DIA + CELL_FIT + RIB_W                # 12.8
CH2_Y = CH1_Y - PITCH                              # チャンネル2軸 y = -55.10
BAR_Y0 = CH2_Y - R_CH - OUT_WALL                   # バー下端 y = -62.75
BAR_Y1 = CASE_BOT_Y + OVERLAP                      # バー上端（壁に食い込み）
BAY_X = CAVITY_LEN / 2                             # 電極面 x = ±24.0
SLOT_X0, SLOT_X1 = BAY_X, BAY_X + SLOT_T
POCK_X1 = SLOT_X1 + POCKET_T
FLOOR_TOP = CASE_BACK_Z + FLOOR_T                  # -2.37
SLOT_Z0 = FLOOR_TOP - SLOT_SINK                    # -3.17
TOWER_TOP = SLOT_Z0 + SLOT_DEPTH                   # 9.33
TOWER_X = BAY_X                                    # タワーは |x| >= 24


def B(x0, x1, y0, y1, z0, z1):
    (x0, x1), (y0, y1), (z0, z1) = sorted((x0, x1)), sorted((y0, y1)), \
        sorted((z0, z1))
    b = box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


def prism(poly, z0, z1):
    m = extrude_polygon(poly, height=z1 - z0)
    m.apply_translation([0, 0, z0])
    return m


def xcyl(r, x0, x1, cy, cz, sections=72):
    c = cylinder(radius=r, height=x1 - x0, sections=sections)
    c.apply_transform(rotation_matrix(np.pi / 2, [0, 1, 0]))
    c.apply_translation([(x0 + x1) / 2, cy, cz])
    return c


def chamfer_x(x0, x1, ey, ez, ch):
    """x 方向に走るエッジ (ey, ez) の 45° 面取りカット"""
    x0, x1 = sorted((x0, x1))
    c = box(extents=[x1 - x0, ch * np.sqrt(2), ch * np.sqrt(2)])
    c.apply_transform(rotation_matrix(np.pi / 4, [1, 0, 0]))
    c.apply_translation([(x0 + x1) / 2, ey, ez])
    return c


def chamfer_y(y0, y1, ex, ez, ch):
    """y 方向に走るエッジ (ex, ez) の 45° 面取りカット"""
    c = box(extents=[ch * np.sqrt(2), y1 - y0, ch * np.sqrt(2)])
    c.apply_transform(rotation_matrix(np.pi / 4, [0, 1, 0]))
    c.apply_translation([ex, (y0 + y1) / 2, ez])
    return c


def union(parts):
    return trimesh.boolean.union(parts, engine='manifold')


def diff(a, parts):
    return trimesh.boolean.difference([a] + parts, engine='manifold')


# ----------------------------------------------------------------------------
# 電池バー
# ----------------------------------------------------------------------------
def build_bar():
    # 平面外形: 上側コーナー r1.2（ケースと面一で接合）、下側コーナー r6
    r_small = 1.2
    base = shapely.ops.unary_union([
        sg.box(-CASE_HX + r_small, BAR_Y0 + CORNER_R,
               CASE_HX - r_small, BAR_Y1 - r_small),
        sg.Point(-(CASE_HX - CORNER_R), BAR_Y0 + CORNER_R)
          .buffer(CORNER_R - r_small, quad_segs=10),
        sg.Point(CASE_HX - CORNER_R, BAR_Y0 + CORNER_R)
          .buffer(CORNER_R - r_small, quad_segs=10),
    ])
    outline = base.buffer(r_small, quad_segs=6, join_style=1)

    # 本体（前面 5.5）+ 電極タワー（|x|>=24 を 9.33 まで）
    bar = prism(outline, CASE_BACK_Z, BAR_FRONT)
    tower_l = prism(sg.box(-CASE_HX - 1, BAR_Y0, -TOWER_X, BAR_Y1)
                    .intersection(outline), CASE_BACK_Z, TOWER_TOP)
    tower_r = prism(sg.box(TOWER_X, BAR_Y0, CASE_HX + 1, BAR_Y1)
                    .intersection(outline), CASE_BACK_Z, TOWER_TOP)
    bar = union([bar, tower_l, tower_r])

    cuts = []

    # ケースとの接合帯（食い込み部）はリムより上に出さない
    cuts.append(B(-CASE_HX - 1, CASE_HX + 1, CASE_BOT_Y - 0.001, BAR_Y1 + EPS,
                  CASE_RIM_Z, TOWER_TOP + 2))

    # --- 電池チャンネル（スナップ保持）---
    for cy in (CH1_Y, CH2_Y):
        cuts.append(xcyl(R_CH, -BAY_X, BAY_X, cy, AXIS_Z))
        # スナップ開口（リップから前面へ垂直壁）
        cuts.append(B(-BAY_X, BAY_X, cy - OPEN_HW, cy + OPEN_HW,
                      LIP_Z, BAR_FRONT + EPS))
        # 開口の面取り（電池を入れやすく）
        cuts.append(B(-BAY_X, BAY_X, cy - OPEN_HW - MOUTH_LEAD,
                      cy + OPEN_HW + MOUTH_LEAD,
                      BAR_FRONT - MOUTH_LEAD, BAR_FRONT + EPS))

    # --- 電極スロット（タワー上端から差し込み、床へ 0.8 食い込み）---
    for cy in (CH1_Y, CH2_Y):  # x- 側: 単体電極 ×2（リード線側）
        cuts.append(B(-SLOT_X1, -SLOT_X0, cy - SLOT_W / 2, cy + SLOT_W / 2,
                      SLOT_Z0, TOWER_TOP + EPS))
    # x+ 側: 直列ブリッジ電極用の幅広スロット
    cuts.append(B(SLOT_X0, SLOT_X1, CH2_Y - SLOT_W / 2, CH1_Y + SLOT_W / 2,
                  SLOT_Z0, TOWER_TOP + EPS))

    # --- 電極背面ポケット（バネの背・ハンダタブの逃げ）---
    for cy in (CH1_Y, CH2_Y):
        cuts.append(B(-POCK_X1, -SLOT_X1 + EPS, cy - SLOT_W / 2,
                      cy + SLOT_W / 2, FLOOR_TOP, TOWER_TOP + EPS))
    cuts.append(B(SLOT_X1 - EPS, POCK_X1, CH2_Y - SLOT_W / 2,
                  CH1_Y + SLOT_W / 2, FLOOR_TOP, TOWER_TOP + EPS))

    # --- 配線ダクト（x- ポケットからトンネルまで、ch1 とケース壁の間）---
    cuts.append(B(-POCK_X1 + 1, -15.5, CH1_Y + 4.0, CH1_Y + R_CH + 0.75,
                  DUCT_Z[0], DUCT_Z[1]))

    # --- 面取り ---
    cuts.append(chamfer_x(-TOWER_X, TOWER_X, BAR_Y0, BAR_FRONT, CHAMFER_F))
    for sx in (-1, 1):
        cuts.append(chamfer_x(sx * TOWER_X, sx * (CASE_HX + 1),
                              BAR_Y0, TOWER_TOP, CHAMFER_T))
        cuts.append(chamfer_y(BAR_Y0 - 1, BAR_Y1 + 1, sx * CASE_HX,
                              TOWER_TOP, CHAMFER_T))
        cuts.append(chamfer_y(BAR_Y0 - 1, BAR_Y1 + 1, sx * CASE_HX,
                              BAR_FRONT, CHAMFER_T))
        # タワー内側の段差エッジ
        cuts.append(chamfer_y(BAR_Y0 - 1, BAR_Y1 + 1, sx * TOWER_X,
                              TOWER_TOP, CHAMFER_T))

    # --- 極性マーク（タワー上面に 0.5 彫り込み）---
    def plus(cx, cy):
        return [B(cx - 1.8, cx + 1.8, cy - 0.6, cy + 0.6,
                  TOWER_TOP - 0.5, TOWER_TOP + EPS),
                B(cx - 0.6, cx + 0.6, cy - 1.8, cy + 1.8,
                  TOWER_TOP - 0.5, TOWER_TOP + EPS)]

    def minus(cx, cy):
        return [B(cx - 1.8, cx + 1.8, cy - 0.6, cy + 0.6,
                  TOWER_TOP - 0.5, TOWER_TOP + EPS)]

    # ch1: + が x+（ブリッジ側）/ ch2: + が x-（リード線側）
    cuts += plus(27.7, CH1_Y) + minus(-27.7, CH1_Y)
    cuts += minus(27.7, CH2_Y) + plus(-27.7, CH2_Y)

    return bar, cuts


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    src = sys.argv[1] if len(sys.argv) > 1 else 'stl/case_fix_2026_original.stl'
    case = trimesh.load(src)
    parts = case.split(only_watertight=False)
    main_body = parts[0]                     # 3906 tris のメインシェル
    others = parts[1:]                       # コーナーポスト・スナップ爪（無改造）
    assert main_body.is_watertight

    bar, cuts = build_bar()

    # 配線トンネル（メインボディの底壁を貫通させる唯一の改造カット）
    tunnel = B(TUNNEL_X[0], TUNNEL_X[1], CH1_Y + OPEN_HW - 0.2, WALL_IN_Y + 1.3,
               TUNNEL_Z[0], TUNNEL_Z[1])

    fused = diff(union([main_body, bar]), cuts + [tunnel])
    print('fused main: watertight =', fused.is_watertight,
          'tris =', len(fused.faces))

    # ポスト・爪はオリジナルのまま同梱（元 STL と同じマルチシェル形式）
    out = trimesh.util.concatenate([fused] + list(others))

    os.makedirs('stl', exist_ok=True)
    path = 'stl/case_battery_2AAA.stl'
    out.export(path)
    b = out.bounds
    print(f'{path}: size = {np.round(out.extents,2).tolist()}, '
          f'bounds z = {b[0][2]:.2f}..{b[1][2]:.2f}')
    fused.export('stl/_fused_main_only.stl')
    print('bar front z =', round(BAR_FRONT, 2), ' tower top z =',
          round(TOWER_TOP, 2), ' total height =',
          round(34.65 + abs(BAR_Y0), 2))
