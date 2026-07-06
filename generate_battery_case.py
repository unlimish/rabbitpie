#!/usr/bin/env python3
"""
Rabbit Pie 単4×3 + ON/OFFスイッチ内蔵ケース ジェネレーター（一体型）
====================================================================

okikata.org の Rabbit Pie 背面ケース (case_fix_2026.stl) の下端を延長し、
単4電池3本（直列 4.5V、ジグザグ配線）と ON/OFF スライドスイッチを
内蔵する「電池バー」を融合した一体型ケース STL を生成します。

- 追加の印刷部品なし（蓋なし・リベットなし）。印刷するのはこのケース1個だけ
- 電池はスナップ保持: チャンネル壁が電池を約220°包み込み、開口を
  電池径より 0.3mm 狭くしてパチンと固定。交換はバネ側へ押してから持ち上げ
- 3本はジグザグ直列: 隣り合う電池を交互に逆向きに挿入し、両端2箇所の
  ブリッジ電極で connect、残り2端だけが外部リード線（→スイッチ→ケース内部）
- 背面はケースと同一平面 → 従来どおり背面を下にした平置き印刷・サポート不要
- 電極タワーに市販の電池ボックス用電極を上から差し込むだけ。フォーク
  プロング付きタブ（実測 20mm×9mm×0.3mm 程度）も、外壁の貫通スリットから
  プロングを突き出して折り曲げロックできる
- 左タワー上部（ケースとの接合部すぐ下）に ON/OFF スライドスイッチの
  ポケットを内蔵（実測 12mm×5mm 程度の市販スライドスイッチを想定）。
  外側からボディごと差し込むフリクションフィットで、奥の細い配線穴で
  電池バー内の配線通路（レースウェイ）とつながる
- リード線はケース底壁を貫通する2本のトンネルで直接内部へ

必要な市販部品:
  - 電池ボックス用電極（幅 ~9-10mm・板厚 ~0.3-1.0mm の一般的な板電極。
    根元に小さな2本足のフォークプロングが付いたタイプ（20mm×9mm×0.3mm
    程度、コイルばね一体型）にも対応: プロングをタワー外壁の貫通スリット
    から外へ突き出して折り曲げロックできる）
      * マイナス側バネ電極 ×1 / プラス側平板電極 ×1（リード線ハンダ付け）
      * 2本連結ブリッジ電極 ×2（無ければ単体電極2枚を銅線で接続）
  - ON/OFF スライドスイッチ ×1（body 12mm×5mm 程度の一般的な小型スライド
    スイッチ。使用中のバッテリーボックスから移植したものでも可）
  - リード線 数本 (AWG24-26)

使い方:
  pip install numpy trimesh manifold3d shapely
  python3 generate_battery_case.py <元ケースSTL>   # 省略時 stl/case_fix_2026_original.stl
"""

import sys
import numpy as np
import trimesh
from trimesh.creation import box, cylinder, extrude_polygon
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
N_CELLS = 3              # 電池本数

# --- ケース実測値 ---
CASE_BACK_Z = -3.97      # 背面（印刷ベッド面）
CASE_RIM_Z = 4.90        # 前面リム
CASE_HX = 29.81          # ケース半幅（X）
CASE_BOT_Y = -34.65      # 底辺外面
WALL_IN_Y = -32.2        # 底壁内面（実測）

# --- 電池バー ---
FLOOR_T = 1.6            # チャンネル床（ベッド側）
TOP_WALL = 17.5          # ケース側〜チャンネル1の間（スイッチ用の間隔を確保）
OUT_WALL = 2.2           # 最終チャンネル外側の壁
RIB_W = 1.9              # チャンネル間ピッチ余裕（ピッチ = Φ + これ）
OVERLAP = 0.65           # ケース底壁への食い込み（融合用）
SLOT_T = 1.0             # 電極スロット厚
SLOT_W = 9.6             # 電極スロット幅（実測 9mm タブ + 0.6 クリアランス）
SLOT_SINK = 0.8          # スロットの床食い込み
POCKET_T = 2.0           # 電極背面ポケット（配線レースウェイ兼用）
SLOT_DEPTH = 12.5        # スロット深さ（電極高さ ~12.5mm まで対応）

# --- フォークプロング固定用スリット（2本足タブ用）---
PRONG_SLOT_W = 6.0       # プロング貫通スリットの幅（Y方向、2本まとめて通す）
PRONG_SLOT_H = 5.0       # プロング貫通スリットの高さ（Z方向、位置合わせの遊び）
PRONG_RECESS_D = 0.6     # 折り曲げたプロングを収める外面側の浅い座ぐり深さ

CORNER_R = 6.0           # バー下側コーナー R
CHAMFER_F = 1.6          # 前面外周エッジの 45° 面取り
CHAMFER_T = 1.2          # タワー上端エッジの 45° 面取り
MOUTH_LEAD = 0.3         # スナップ開口の面取り（入れやすさ）

# --- ON/OFF スライドスイッチ（実測 ~12mm×5mm 相当）---
SWITCH_L = 13.0          # ポケット長さ（Y方向、スイッチ長 12mm + 1 クリアランス）
SWITCH_H = 5.6           # ポケット高さ（Z方向、スイッチ幅 5mm + 0.6 クリアランス）
SWITCH_D = 4.6           # ボディ収納深さ（X方向、フリクションフィット）
SWITCH_Z0 = -0.8         # ポケット下端 Z
SWITCH_MARGIN = 2.0      # スイッチポケット〜ケース接合部の余白
SWITCH_WIRE_W = 4.0      # ボディ奥の配線用の細い貫通穴（幅）
SWITCH_WIRE_H = 3.0      # 同（高さ）

# --- 配線トンネル（左右2本、ケース底壁を貫通）---
TUNNEL_XL = (-19.0, -16.0)
TUNNEL_XR = (16.0, 19.0)
TUNNEL_Y = (-34.4, -30.9)          # 背面プレート上面〜基板ハンダ面の間の帯
TUNNEL_Z = (-2.8, -0.6)

EPS = 0.05

# ----------------------------------------------------------------------------
# 導出値
# ----------------------------------------------------------------------------
R_CH = (CELL_DIA + CELL_FIT) / 2                   # チャンネル半径 5.45
AXIS_Z = CASE_BACK_Z + FLOOR_T + R_CH              # 電池軸 z = 3.08
OPEN_HW = (CELL_DIA - SNAP_PINCH) / 2              # スナップ開口半幅 5.1
LIP_Z = AXIS_Z + np.sqrt(R_CH**2 - OPEN_HW**2)     # リップ z
BAR_FRONT = LIP_Z + 0.5                            # バー前面
PITCH = CELL_DIA + CELL_FIT + RIB_W                # チャンネルピッチ

CH_Y = [CASE_BOT_Y - TOP_WALL - R_CH - i * PITCH for i in range(N_CELLS)]
CH1_Y, CH2_Y, CH3_Y = CH_Y                         # 可読性のためのエイリアス

BAR_Y0 = CH_Y[-1] - R_CH - OUT_WALL                # バー下端
BAR_Y1 = CASE_BOT_Y + OVERLAP                      # バー上端（壁に食い込み）
BAY_X = CAVITY_LEN / 2                             # 電極面 x = ±24.0
SLOT_X0, SLOT_X1 = BAY_X, BAY_X + SLOT_T
POCK_X1 = SLOT_X1 + POCKET_T
FLOOR_TOP = CASE_BACK_Z + FLOOR_T                  # -2.37
SLOT_Z0 = FLOOR_TOP - SLOT_SINK                    # -3.17
TOWER_TOP = SLOT_Z0 + SLOT_DEPTH                   # 9.33
TOWER_X = BAY_X                                    # タワーは |x| >= 24

SWITCH_Y1 = BAR_Y1 - SWITCH_MARGIN
SWITCH_Y0 = SWITCH_Y1 - SWITCH_L
SWITCH_Z1 = SWITCH_Z0 + SWITCH_H
SWITCH_CY = (SWITCH_Y0 + SWITCH_Y1) / 2


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


def prong_feature(sign, cy):
    """タワー外壁を貫通するフォークプロング用スリット＋外面の浅い座ぐり。
    電極タブを差し込んだ後、根元の2本足プロングをここから外へ突き出し、
    外面に沿って折り曲げれば抜け止めロックになる。"""
    slot = B(sign * POCK_X1, sign * (CASE_HX + 1),
             cy - PRONG_SLOT_W / 2, cy + PRONG_SLOT_W / 2,
             AXIS_Z - PRONG_SLOT_H / 2, AXIS_Z + PRONG_SLOT_H / 2)
    recess = B(sign * (CASE_HX - PRONG_RECESS_D), sign * (CASE_HX + 1),
               cy - PRONG_SLOT_W / 2 - 2, cy + PRONG_SLOT_W / 2 + 2,
               AXIS_Z - PRONG_SLOT_H / 2 - 2, AXIS_Z + PRONG_SLOT_H / 2 + 2)
    return [slot, recess]


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

    # --- 電池チャンネル（スナップ保持、3本）---
    for cy in CH_Y:
        cuts.append(xcyl(R_CH, -BAY_X, BAY_X, cy, AXIS_Z))
        # スナップ開口（リップから前面へ垂直壁）
        cuts.append(B(-BAY_X, BAY_X, cy - OPEN_HW, cy + OPEN_HW,
                      LIP_Z, BAR_FRONT + EPS))
        # 開口の面取り（電池を入れやすく）
        cuts.append(B(-BAY_X, BAY_X, cy - OPEN_HW - MOUTH_LEAD,
                      cy + OPEN_HW + MOUTH_LEAD,
                      BAR_FRONT - MOUTH_LEAD, BAR_FRONT + EPS))

    # --- 電極スロット（ジグザグ直列配線）---
    # x-タワー: ch1 単体リード線スロット／ch2-ch3 ブリッジスロット
    # x+タワー: ch1-ch2 ブリッジスロット／ch3 単体リード線スロット
    def slot_cut(sign, y0, y1):
        x0 = sign * SLOT_X0
        x1 = sign * SLOT_X1
        return B(x0, x1, y0 - SLOT_W / 2, y1 + SLOT_W / 2,
                 SLOT_Z0, TOWER_TOP + EPS)

    def pocket_cut(sign, y0, y1):
        x0 = sign * SLOT_X1
        x1 = sign * POCK_X1
        return B(x0, x1, y0 - SLOT_W / 2, y1 + SLOT_W / 2,
                 FLOOR_TOP, TOWER_TOP + EPS)

    cuts.append(slot_cut(-1, CH1_Y, CH1_Y))
    cuts.append(slot_cut(-1, CH3_Y, CH2_Y))
    cuts.append(slot_cut(1, CH1_Y, CH2_Y))
    cuts.append(slot_cut(1, CH3_Y, CH3_Y))
    cuts.append(pocket_cut(-1, CH1_Y, CH1_Y))
    cuts.append(pocket_cut(-1, CH3_Y, CH2_Y))
    cuts.append(pocket_cut(1, CH1_Y, CH2_Y))
    cuts.append(pocket_cut(1, CH3_Y, CH3_Y))

    # --- 配線レースウェイ（左右タワー内、ケース接合部から最終チャンネル
    #     手前まで連続した溝。電極ポケット同士・スイッチ・トンネルを繋ぐ）---
    for sign in (-1, 1):
        x0 = sign * POCK_X1
        x1 = sign * SLOT_X1
        cuts.append(B(x0, x1, BAR_Y1 - EPS, CH_Y[-1] - SLOT_W / 2 - EPS,
                      FLOOR_TOP, TOWER_TOP + EPS))

    # --- フォークプロング固定スリット（左右タワー×各チャンネル）---
    for cy in CH_Y:
        cuts += prong_feature(-1, cy)
        cuts += prong_feature(1, cy)

    # --- ON/OFF スライドスイッチ ポケット（左タワー上部）---
    #     外側からボディごとフリクションフィットで挿入。奥は細い配線穴で
    #     レースウェイへつながる（ボディが通り抜けない肩=段差になる）
    cuts.append(B(-(CASE_HX + 1), -(CASE_HX - SWITCH_D),
                  SWITCH_Y0, SWITCH_Y1, SWITCH_Z0, SWITCH_Z1))
    cuts.append(B(-(CASE_HX - SWITCH_D + EPS), -SLOT_X1,
                  SWITCH_CY - SWITCH_WIRE_W / 2, SWITCH_CY + SWITCH_WIRE_W / 2,
                  AXIS_Z - SWITCH_WIRE_H / 2, AXIS_Z + SWITCH_WIRE_H / 2))

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

    # ジグザグ配線: ch1 は x+ が+/x- が-、ch2 は逆(x- が+/x+ が-)、
    # ch3 は ch1 と同じ向き
    cuts += plus(27.7, CH1_Y) + minus(-27.7, CH1_Y)
    cuts += minus(27.7, CH2_Y) + plus(-27.7, CH2_Y)
    cuts += plus(27.7, CH3_Y) + minus(-27.7, CH3_Y)

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

    # 配線トンネル（メインボディの底壁を貫通させる、左右2本）
    tunnel_l = B(TUNNEL_XL[0], TUNNEL_XL[1], TUNNEL_Y[0], TUNNEL_Y[1],
                 TUNNEL_Z[0], TUNNEL_Z[1])
    tunnel_r = B(TUNNEL_XR[0], TUNNEL_XR[1], TUNNEL_Y[0], TUNNEL_Y[1],
                 TUNNEL_Z[0], TUNNEL_Z[1])

    fused = diff(union([main_body, bar]), cuts + [tunnel_l, tunnel_r])
    print('fused main: watertight =', fused.is_watertight,
          'tris =', len(fused.faces))

    # ポスト・爪はオリジナルのまま同梱（元 STL と同じマルチシェル形式）
    out = trimesh.util.concatenate([fused] + list(others))

    os.makedirs('stl', exist_ok=True)
    path = 'stl/case_battery_3AAA_switch.stl'
    out.export(path)
    b = out.bounds
    print(f'{path}: size = {np.round(out.extents,2).tolist()}, '
          f'bounds z = {b[0][2]:.2f}..{b[1][2]:.2f}')
    fused.export('stl/_fused_main_only.stl')
    print('bar front z =', round(BAR_FRONT, 2), ' tower top z =',
          round(TOWER_TOP, 2), ' total height =',
          round(34.65 + abs(BAR_Y0), 2))
    print('channels Y =', [round(y, 2) for y in CH_Y])
    print('switch pocket Y =', round(SWITCH_Y0, 2), '..', round(SWITCH_Y1, 2),
          ' Z =', round(SWITCH_Z0, 2), '..', round(SWITCH_Z1, 2))
