#!/usr/bin/env python3
"""
Rabbit Pie 単4×3 電池ケース ジェネレーター（一体型、2バリエーション）
======================================================================

okikata.org の Rabbit Pie 背面ケース (case_fix_2026.stl) の下端を延長し、
単4電池3本（直列 4.5V、ジグザグ配線）を内蔵する「電池バー」を融合した
一体型ケース STL を生成します。2つのバリエーションを出力します:

  1. stl/case_3AAA_basic.stl
     電池3本 + ON/OFF電源スイッチのみ。コンパクト版
  2. stl/case_3AAA_speaker.stl + stl/speaker_lid.stl
     電池3本 + ON/OFF電源スイッチ + スピーカー用ミュートスイッチ + スピーカー
     ポケット（着脱式リッド別部品）。スピーカーはリッドを外して脱着できる

共通の特徴:
- 電池はスナップ保持: チャンネル壁が電池を約220°包み込み、開口を
  電池径より 0.3mm 狭くしてパチンと固定。交換はバネ側へ押してから持ち上げ
- 3本はジグザグ直列: 隣り合う電池を交互に逆向きに挿入し、両端2箇所の
  ブリッジ電極で接続、残り2端だけが外部リード線
- 背面はケースと同一平面 → 従来どおり背面を下にした平置き印刷・サポート不要
- 電極タワーに市販の電池ボックス用電極を上から差し込むだけ。フォーク
  プロング付きタブ（実測 20mm×9mm×0.3mm 程度）も、外壁の貫通スリットから
  プロングを突き出して折り曲げロックできる
- リード線はケース底壁を貫通するトンネルで直接内部へ

speaker バリアントのみの追加要素:
- 左タワー上部に ON/OFF 電源スイッチ、右タワー上部にスピーカー用ミュート
  スイッチ（どちらも実測 12mm×5mm 程度の市販小型スライドスイッチを想定）
- バー中央部に小型スピーカー（27×17mm 程度）用の深いポケット。前面は
  着脱式リッド（別部品、グリル穴付き）で覆う。リッドは浅いリベート
  （額縁状の段差）に上下の保持リップでスナップイン
- イヤホンジャックに挿入検知機能が無い場合の代替として、上記ミュート
  スイッチで手動でスピーカーを消音する運用を想定

必要な市販部品:
  - 電池ボックス用電極（幅 ~9-10mm・板厚 ~0.3-1.0mm の一般的な板電極。
    根元に小さな2本足のフォークプロングが付いたタイプにも対応）
      * マイナス側バネ電極 ×1 / プラス側平板電極 ×1（リード線ハンダ付け）
      * 2本連結ブリッジ電極 ×2（無ければ単体電極2枚を銅線で接続）
  - ON/OFF スライドスイッチ ×1（basic）または ×2（speaker、電源+ミュート）
  - 小型スピーカー ×1（speaker のみ。27×17mm 程度、8Ω 0.5-1W）
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
TOP_WALL_BASIC = 17.5    # ケース側〜チャンネル1の間（電源スイッチのみ用）
TOP_WALL_SPEAKER = 22.0  # 同（電源+ミュートスイッチ＋スピーカー用）
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

# --- ON/OFF スライドスイッチ（実測 ~12mm×5mm 相当。電源用/ミュート用共通）---
SWITCH_L = 13.0          # ポケット長さ（Y方向、スイッチ長 12mm + 1 クリアランス）
SWITCH_H = 5.6           # ポケット高さ（Z方向、スイッチ幅 5mm + 0.6 クリアランス）
SWITCH_D = 4.6           # ボディ収納深さ（X方向、フリクションフィット）
SWITCH_Z0 = -0.8         # ポケット下端 Z
SWITCH_MARGIN = 2.0      # スイッチポケット〜ケース接合部の余白
SWITCH_WIRE_W = 4.0      # ボディ奥の配線用の細い貫通穴（幅）
SWITCH_WIRE_H = 3.0      # 同（高さ）

# --- 小型スピーカー（実測 ~27mm×17mm 相当の楕円形マイクロスピーカー）---
SPK_HW = 15.0            # ポケット半幅 X（スピーカー長辺27mm + 3クリアランス）
SPK_HH = 9.0             # ポケット半高 Y（スピーカー短辺17mm + 1クリアランス）
SPK_BODY_D = 4.5         # ボディ収納深さ Z（前面リッドの奥）
SPK_MARGIN = 2.0         # スピーカーポケット〜隣接構造の余白

# --- スピーカー着脱式リッド（別部品、スナップイン）---
LID_T = 1.4              # リッド厚み
LID_CLR = 0.3            # リッド外周クリアランス
LIP_D = 1.0              # 保持リップの前後方向の厚み
LIP_ENGAGE = 2.0         # 保持リップが上下端にかぶる幅
GRILLE_HOLE_D = 2.0      # グリルの音穴直径
GRILLE_PITCH = 4.0       # 音穴の間隔（格子状）
GRILLE_INSET = 2.5       # リッド外周から音穴パターンまでの余白（保持強度確保）
SPEAKER_WIRE_W = 4.0     # 配線ダクトの幅
SPEAKER_WIRE_H = 3.0     # 配線ダクトの高さ

# --- 配線トンネル（左右2本、ケース底壁を貫通。ch1/ch3 の外部リード線は
#     どちらも必ずどこかのトンネルへ抜ける必要があるため両方とも常設）---
TUNNEL_XL = (-19.0, -16.0)
TUNNEL_XR = (16.0, 19.0)
TUNNEL_Y = (-34.4, -30.9)          # 背面プレート上面〜基板ハンダ面の間の帯
TUNNEL_Z = (-2.8, -0.6)

EPS = 0.05

# ----------------------------------------------------------------------------
# 導出値（電池・スロット関連は共通、TOP_WALL に依存しない）
# ----------------------------------------------------------------------------
R_CH = (CELL_DIA + CELL_FIT) / 2                   # チャンネル半径 5.45
AXIS_Z = CASE_BACK_Z + FLOOR_T + R_CH              # 電池軸 z = 3.08
OPEN_HW = (CELL_DIA - SNAP_PINCH) / 2              # スナップ開口半幅 5.1
LIP_Z = AXIS_Z + np.sqrt(R_CH**2 - OPEN_HW**2)     # リップ z
BAR_FRONT = LIP_Z + 0.5                            # バー前面
PITCH = CELL_DIA + CELL_FIT + RIB_W                # チャンネルピッチ

BAY_X = CAVITY_LEN / 2                             # 電極面 x = ±24.0
SLOT_X0, SLOT_X1 = BAY_X, BAY_X + SLOT_T
POCK_X1 = SLOT_X1 + POCKET_T
FLOOR_TOP = CASE_BACK_Z + FLOOR_T                  # -2.37
SLOT_Z0 = FLOOR_TOP - SLOT_SINK                    # -3.17
TOWER_TOP = SLOT_Z0 + SLOT_DEPTH                   # 9.33
TOWER_X = BAY_X                                    # タワーは |x| >= 24
THIN_DUCT_Z = (FLOOR_TOP, FLOOR_TOP + 3.0)          # 最適化版の細い配線ダクト高さ


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


def switch_pocket_cuts(sign, cy):
    """ON/OFF スライドスイッチ ポケット（左=電源用 sign=-1／右=ミュート用 sign=1）。
    外側からボディごとフリクションフィットで挿入。奥は細い配線穴で
    レースウェイへつながる（ボディが通り抜けない肩=段差になる）"""
    y0, y1 = cy - SWITCH_L / 2, cy + SWITCH_L / 2
    z0, z1 = SWITCH_Z0, SWITCH_Z0 + SWITCH_H
    body = B(sign * (CASE_HX + 1), sign * (CASE_HX - SWITCH_D), y0, y1, z0, z1)
    wire = B(sign * (CASE_HX - SWITCH_D + EPS), sign * SLOT_X1,
             cy - SWITCH_WIRE_W / 2, cy + SWITCH_WIRE_W / 2,
             THIN_DUCT_Z[0], THIN_DUCT_Z[0] + SWITCH_WIRE_H)
    return [body, wire]


def grille_holes(cx, cy, w, h, z0, z1):
    """矩形領域内に格子状の音穴を並べる（外周 GRILLE_INSET は保持リムとして残す）"""
    hw = w / 2 - GRILLE_INSET
    hh = h / 2 - GRILLE_INSET
    nx = max(1, int(2 * hw // GRILLE_PITCH) + 1)
    ny = max(1, int(2 * hh // GRILLE_PITCH) + 1)
    xs = np.linspace(cx - hw, cx + hw, nx) if nx > 1 else [cx]
    ys = np.linspace(cy - hh, cy + hh, ny) if ny > 1 else [cy]
    holes = []
    for hx in xs:
        for hy in ys:
            c = cylinder(radius=GRILLE_HOLE_D / 2, height=z1 - z0 + 2 * EPS,
                         sections=24)
            c.apply_translation([hx, hy, (z0 + z1) / 2])
            holes.append(c)
    return holes


def union(parts):
    return trimesh.boolean.union(parts, engine='manifold')


def diff(a, parts):
    return trimesh.boolean.difference([a] + parts, engine='manifold')


# ----------------------------------------------------------------------------
# 電池バー本体
# ----------------------------------------------------------------------------
def build_bar(with_speaker, optimized=False):
    top_wall = TOP_WALL_SPEAKER if with_speaker else TOP_WALL_BASIC
    ch_y = [CASE_BOT_Y - top_wall - R_CH - i * PITCH for i in range(N_CELLS)]
    ch1_y, ch2_y, ch3_y = ch_y
    bar_y0 = ch_y[-1] - R_CH - OUT_WALL
    bar_y1 = CASE_BOT_Y + OVERLAP

    switch_y1 = bar_y1 - SWITCH_MARGIN
    switch_cy = switch_y1 - SWITCH_L / 2

    # 平面外形: 上側コーナー r1.2（ケースと面一で接合）、下側コーナー r6
    r_small = 1.2
    base = shapely.ops.unary_union([
        sg.box(-CASE_HX + r_small, bar_y0 + CORNER_R,
               CASE_HX - r_small, bar_y1 - r_small),
        sg.Point(-(CASE_HX - CORNER_R), bar_y0 + CORNER_R)
          .buffer(CORNER_R - r_small, quad_segs=10),
        sg.Point(CASE_HX - CORNER_R, bar_y0 + CORNER_R)
          .buffer(CORNER_R - r_small, quad_segs=10),
    ])
    outline = base.buffer(r_small, quad_segs=6, join_style=1)

    # 本体（前面 5.5）+ 電極タワー（|x|>=24 を 9.33 まで）
    bar = prism(outline, CASE_BACK_Z, BAR_FRONT)
    tower_l = prism(sg.box(-CASE_HX - 1, bar_y0, -TOWER_X, bar_y1)
                    .intersection(outline), CASE_BACK_Z, TOWER_TOP)
    tower_r = prism(sg.box(TOWER_X, bar_y0, CASE_HX + 1, bar_y1)
                    .intersection(outline), CASE_BACK_Z, TOWER_TOP)
    bar = union([bar, tower_l, tower_r])

    cuts = []

    # ケースとの接合帯（食い込み部）はリムより上に出さない
    cuts.append(B(-CASE_HX - 1, CASE_HX + 1, CASE_BOT_Y - 0.001, bar_y1 + EPS,
                  CASE_RIM_Z, TOWER_TOP + 2))

    # --- 電池チャンネル（スナップ保持、3本）---
    for cy in ch_y:
        cuts.append(xcyl(R_CH, -BAY_X, BAY_X, cy, AXIS_Z))
        cuts.append(B(-BAY_X, BAY_X, cy - OPEN_HW, cy + OPEN_HW,
                      LIP_Z, BAR_FRONT + EPS))
        cuts.append(B(-BAY_X, BAY_X, cy - OPEN_HW - MOUTH_LEAD,
                      cy + OPEN_HW + MOUTH_LEAD,
                      BAR_FRONT - MOUTH_LEAD, BAR_FRONT + EPS))

    # --- 電極スロット（ジグザグ直列配線）---
    # x-タワー: ch1 単体リード線スロット／ch2-ch3 ブリッジスロット
    # x+タワー: ch1-ch2 ブリッジスロット／ch3 単体リード線スロット
    def slot_cut(sign, ya, yb):
        y0, y1 = sorted((ya, yb))
        x0, x1 = sign * SLOT_X0, sign * SLOT_X1
        return B(x0, x1, y0 - SLOT_W / 2, y1 + SLOT_W / 2,
                 SLOT_Z0, TOWER_TOP + EPS)

    def pocket_cut(sign, ya, yb):
        y0, y1 = sorted((ya, yb))
        x0, x1 = sign * SLOT_X1, sign * POCK_X1
        return B(x0, x1, y0 - SLOT_W / 2, y1 + SLOT_W / 2,
                 FLOOR_TOP, TOWER_TOP + EPS)

    cuts.append(slot_cut(-1, ch1_y, ch1_y))
    cuts.append(slot_cut(-1, ch3_y, ch2_y))
    cuts.append(slot_cut(1, ch1_y, ch2_y))
    cuts.append(slot_cut(1, ch3_y, ch3_y))
    cuts.append(pocket_cut(-1, ch1_y, ch1_y))
    cuts.append(pocket_cut(-1, ch3_y, ch2_y))
    cuts.append(pocket_cut(1, ch1_y, ch2_y))
    cuts.append(pocket_cut(1, ch3_y, ch3_y))

    if not optimized:
        # --- 配線レースウェイ（左右タワー内、ケース接合部から最終チャンネル
        #     手前まで連続した溝。電極ポケット同士・スイッチ・トンネルを繋ぐ）---
        for sign in (-1, 1):
            x0, x1 = sign * POCK_X1, sign * SLOT_X1
            cuts.append(B(x0, x1, bar_y1 - EPS, ch_y[-1] - SLOT_W / 2 - EPS,
                          FLOOR_TOP, TOWER_TOP + EPS))
    else:
        # --- 印刷最適化: レースウェイ全体を埋め、必要な配線経路だけを
        #     細いダクト（THIN_DUCT_Z の高さのみ）で確保する。内部の
        #     大きな空洞を減らし、印刷時の天井ブリッジ量を削減する ---
        dz0, dz1 = THIN_DUCT_Z
        # x- タワー: ch1（外部リード線）〜電源スイッチ
        cuts.append(B(-POCK_X1, -SLOT_X1, ch1_y + SLOT_W / 2 - EPS,
                      switch_cy + SWITCH_WIRE_W / 2, dz0, dz1))
        # x+ タワー: ch3（外部リード線）〜ケース接合部（右トンネルへ）
        cuts.append(B(SLOT_X1, POCK_X1, ch3_y + SLOT_W / 2 - EPS,
                      bar_y1 - EPS, dz0, dz1))

    # --- フォークプロング固定スリット（左右タワー×各チャンネル）---
    for cy in ch_y:
        cuts += prong_feature(-1, cy)
        cuts += prong_feature(1, cy)

    # --- ON/OFF 電源スイッチ ポケット（左タワー上部）---
    cuts += switch_pocket_cuts(-1, switch_cy)

    speaker_cy = None
    if with_speaker:
        # --- ミュートスイッチ ポケット（右タワー上部、電源スイッチと対称）---
        cuts += switch_pocket_cuts(1, switch_cy)

        # --- 小型スピーカー ポケット（バー中央部、着脱式リッドで覆う）---
        gap_y1 = bar_y1 - SPK_MARGIN
        gap_y0 = ch1_y + R_CH + SPK_MARGIN
        speaker_cy = (gap_y0 + gap_y1) / 2
        assert gap_y1 - gap_y0 >= 2 * SPK_HH, \
            f'gap zone too short for speaker: {gap_y1-gap_y0:.1f} < {2*SPK_HH}'

        z_lid_in = BAR_FRONT - LID_T
        z_body_bot = z_lid_in - SPK_BODY_D

        # 深いボディポケット（前面のリッド厚み分は含まない）
        cuts.append(B(-SPK_HW, SPK_HW, speaker_cy - SPK_HH, speaker_cy + SPK_HH,
                      z_body_bot, z_lid_in + EPS))
        # リッドが収まるリベート（外周 LID_CLR 分だけ大きい浅い段差）
        cuts.append(B(-SPK_HW - LID_CLR, SPK_HW + LID_CLR,
                      speaker_cy - SPK_HH - LID_CLR, speaker_cy + SPK_HH + LID_CLR,
                      z_lid_in, BAR_FRONT - LIP_D))
        # 上下端の保持リップを除いた中央部は前面まで貫通
        cuts.append(B(-SPK_HW - LID_CLR, SPK_HW + LID_CLR,
                      speaker_cy - SPK_HH + LIP_ENGAGE,
                      speaker_cy + SPK_HH - LIP_ENGAGE,
                      BAR_FRONT - LIP_D, BAR_FRONT + EPS))

        # 配線ダクト（L字: スピーカー左端 -> 左トンネルの x 位置 -> トンネルまで）
        duct_z0 = z_body_bot + 1.0
        duct_z1 = duct_z0 + SPEAKER_WIRE_H
        cuts.append(B(TUNNEL_XL[0] - 1.5, -SPK_HW - LID_CLR + EPS,
                      speaker_cy - SPEAKER_WIRE_W / 2,
                      speaker_cy + SPEAKER_WIRE_W / 2, duct_z0, duct_z1))
        cuts.append(B(TUNNEL_XL[0] - 1.5, TUNNEL_XL[1] + 1.5,
                      TUNNEL_Y[1] - EPS, speaker_cy + SPEAKER_WIRE_W / 2,
                      duct_z0, duct_z1))

    # --- 面取り ---
    cuts.append(chamfer_x(-TOWER_X, TOWER_X, bar_y0, BAR_FRONT, CHAMFER_F))
    for sx in (-1, 1):
        cuts.append(chamfer_x(sx * TOWER_X, sx * (CASE_HX + 1),
                              bar_y0, TOWER_TOP, CHAMFER_T))
        cuts.append(chamfer_y(bar_y0 - 1, bar_y1 + 1, sx * CASE_HX,
                              TOWER_TOP, CHAMFER_T))
        cuts.append(chamfer_y(bar_y0 - 1, bar_y1 + 1, sx * CASE_HX,
                              BAR_FRONT, CHAMFER_T))
        cuts.append(chamfer_y(bar_y0 - 1, bar_y1 + 1, sx * TOWER_X,
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
    cuts += plus(27.7, ch1_y) + minus(-27.7, ch1_y)
    cuts += minus(27.7, ch2_y) + plus(-27.7, ch2_y)
    cuts += plus(27.7, ch3_y) + minus(-27.7, ch3_y)

    info = dict(ch_y=ch_y, bar_y0=bar_y0, bar_y1=bar_y1,
                switch_cy=switch_cy, speaker_cy=speaker_cy)
    return bar, cuts, info


# ----------------------------------------------------------------------------
# スピーカー着脱式リッド（別部品）
# ----------------------------------------------------------------------------
def build_speaker_lid(speaker_cy):
    """段付き形状: 中央部は前面(BAR_FRONT)まで達し、上下端の LIP_ENGAGE 幅
    だけ LIP_D 分薄くして、ケース側の保持リップの下に滑り込むようにする。"""
    hw, hh = SPK_HW - LID_CLR / 2, SPK_HH - LID_CLR / 2
    z0, z1 = BAR_FRONT - LID_T, BAR_FRONT
    center = B(-hw, hw, speaker_cy - (hh - LIP_ENGAGE),
              speaker_cy + (hh - LIP_ENGAGE), z0, z1)
    edge_top = B(-hw, hw, speaker_cy + (hh - LIP_ENGAGE), speaker_cy + hh,
                z0, z1 - LIP_D)
    edge_bot = B(-hw, hw, speaker_cy - hh, speaker_cy - (hh - LIP_ENGAGE),
                z0, z1 - LIP_D)
    lid = union([center, edge_top, edge_bot])
    holes = grille_holes(0.0, speaker_cy, 2 * hw, 2 * (hh - LIP_ENGAGE),
                         z0 - EPS, z1 + EPS)
    lid = diff(lid, holes)
    # 印刷姿勢: 外面(z1側)を下にする
    lid.apply_translation([0, -speaker_cy, 0])
    lid.apply_transform(rotation_matrix(np.pi, [1, 0, 0]))
    lid.apply_translation([0, 0, -lid.bounds[0][2]])
    return lid


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def build_case(main_body, others, with_speaker, optimized=False):
    bar, cuts, info = build_bar(with_speaker, optimized=optimized)

    # ch1(x-)・ch3(x+) の外部リード線は必ずどちらかのトンネルへ抜ける必要が
    # あるため、電源スイッチ/スピーカーの有無にかかわらず左右とも常設する
    tunnels = [
        B(TUNNEL_XL[0], TUNNEL_XL[1], TUNNEL_Y[0], TUNNEL_Y[1],
          TUNNEL_Z[0], TUNNEL_Z[1]),
        B(TUNNEL_XR[0], TUNNEL_XR[1], TUNNEL_Y[0], TUNNEL_Y[1],
          TUNNEL_Z[0], TUNNEL_Z[1]),
    ]

    fused = diff(union([main_body, bar]), cuts + tunnels)
    out = trimesh.util.concatenate([fused] + list(others))
    return out, fused, info


if __name__ == '__main__':
    import os
    src = sys.argv[1] if len(sys.argv) > 1 else 'stl/case_fix_2026_original.stl'
    case = trimesh.load(src)
    parts = case.split(only_watertight=False)
    main_body = parts[0]                     # 3906 tris のメインシェル
    others = parts[1:]                       # コーナーポスト・スナップ爪（無改造）
    assert main_body.is_watertight

    os.makedirs('stl', exist_ok=True)

    out, fused, info = build_case(main_body, others, with_speaker=False)
    print('basic: watertight =', fused.is_watertight, 'tris =', len(fused.faces))
    out.export('stl/case_3AAA_basic.stl')
    print(f'  size = {np.round(out.extents,2).tolist()}')
    print('  channels Y =', [round(y, 2) for y in info['ch_y']])

    out_opt, fused_opt, info_opt = build_case(main_body, others,
                                              with_speaker=False, optimized=True)
    print('basic (print-optimized, no raceway): watertight =',
          fused_opt.is_watertight, 'tris =', len(fused_opt.faces))
    out_opt.export('stl/case_3AAA_basic_optimized.stl')
    print(f'  size = {np.round(out_opt.extents,2).tolist()}')

    out2, fused2, info2 = build_case(main_body, others, with_speaker=True)
    print('speaker: watertight =', fused2.is_watertight, 'tris =', len(fused2.faces))
    out2.export('stl/case_3AAA_speaker.stl')
    print(f'  size = {np.round(out2.extents,2).tolist()}')
    print('  channels Y =', [round(y, 2) for y in info2['ch_y']])
    print('  power switch cy =', round(info2['switch_cy'], 2))
    print('  speaker cy =', round(info2['speaker_cy'], 2))

    lid = build_speaker_lid(info2['speaker_cy'])
    lid.export('stl/speaker_lid.stl')
    print('  speaker_lid size =', np.round(lid.extents, 2).tolist())

    fused2.export('stl/_fused_main_only.stl')
