import numpy as np
import matplotlib.pyplot as plt
import trimesh

# tamentai: (2, 2, 2, 5, 3)
tamentai = np.zeros((2, 2, 3, 5, 3), dtype=float)
# hesa: (2, 6, 3)  2リング × 各6点 × xyz
hesa = np.zeros((2, 2, 6, 3), dtype=float)

def santobun(point_set):
    p1 = point_set[0]
    p2 = point_set[1]

    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]

    x2 = p2[0]
    y2 = p2[1]
    z2 = p2[2]

    return (2*x1+x2)/3.0, (2*y1+y2)/3.0, (2*z1+z2)/3.0   

def santobun2(point_set):
    """
    2点 p1, p2 から線分を3等分する 2点を返す関数
    戻り値:
      (p1側1/3, p2側1/3) の 2点 ＝ 6個の値
    """
    p1 = point_set[0]
    p2 = point_set[1]

    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]

    x2 = p2[0]
    y2 = p2[1]
    z2 = p2[2]

    # p1 から 1/3 の点, p2 から 1/3 の点
    return (2*x1+x2)/3.0, (2*y1+y2)/3.0, (2*z1+z2)/3.0, \
           (x1+2*x2)/3.0, (y1+2*y2)/3.0, (z1+2*z2)/3.0

# === 共通化してもよいが、わかりやすさのため tamentai 用と hesa 用を分ける ===

def build_vertices_and_faces_from_tamentai(triangle_index_list):
    """
    triangle_index_list:
        [ ((i0,j0,k0,l0), (i1,j1,k1,l1), (i2,j2,k2,l2)),  ... ]

    tamentai[i,j,k,l,:] を参照して、
      - vertices: (N,3)
      - faces   : (M,3)
    を返す。
    """
    vertices = []
    index_to_vid = {}  # (i,j,k,l) -> vertex id
    faces = []

    for tri in triangle_index_list:
        face_vids = []
        for idx4 in tri:
            if idx4 not in index_to_vid:
                x, y, z = tamentai[idx4[0], idx4[1], idx4[2], idx4[3], :]
                vid = len(vertices)
                vertices.append([x, y, z])
                index_to_vid[idx4] = vid
            face_vids.append(index_to_vid[idx4])
        faces.append(face_vids)

    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=np.int64)
    return vertices, faces

def export_stl_from_triangle_indices(triangle_index_list, filename="output.stl"):
    """
    tamentai のインデックスで指定した三角形リストから STL を出力する関数。
    """
    vertices, faces = build_vertices_and_faces_from_tamentai(triangle_index_list)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(filename)
    print(f"STL ファイルを書き出しました: {filename}")

# --- ここから hesa 用 ---
def fetch_hesa_point(idx_tuple):
    """
    idx_tuple:
       (ring, layer, idx)  または  (ring, idx) を許容。
       後者の場合 layer=0 とみなす。
    """
    if len(idx_tuple) == 3:
        r, layer, i = idx_tuple
    elif len(idx_tuple) == 2:
        r, i = idx_tuple
        layer = 0
    else:
        raise ValueError("idx_tuple は長さ2か3である必要があります")

    return hesa[r, layer, i, :]

def build_vertices_and_faces_from_hesa(triangle_index_list):
    """
    triangle_index_list:
        [ (idx_tuple0, idx_tuple1, idx_tuple2), ... ]
    idx_tuple は (ring, idx) でも (a,b,c,d) でもよい。

    すべて fetch_hesa_point が吸収する。
    """
    vertices = []
    index_to_vid = {}  # idx_tuple -> vertex id
    faces = []

    for tri in triangle_index_list:
        face_vids = []
        for idx_tuple in tri:
            if idx_tuple not in index_to_vid:
                x, y, z = fetch_hesa_point(idx_tuple)
                vid = len(vertices)
                vertices.append([x, y, z])
                index_to_vid[idx_tuple] = vid
            face_vids.append(index_to_vid[idx_tuple])
        faces.append(face_vids)

    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=np.int64)
    return vertices, faces

def export_stl_from_hesa_triangle_indices(triangle_index_list, filename="hesa_output.stl"):
    """
    hesa のインデックスで指定した三角形リストから STL を出力する関数。
    """
    vertices, faces = build_vertices_and_faces_from_hesa(triangle_index_list)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(filename)
    print(f"hesa から STL ファイルを書き出しました: {filename}")

def y_θ(x, y, z, a):

    x1 = x * np.cos(np.radians(a)) + z * np.sin(np.radians(a))
    y1 = y
    z1 = -x*np.sin(np.radians(a)) + z * np.cos(np.radians(a))

    return x1, y1, z1

# --- メイン処理 ---
def main():
    global tamentai, hesa

    # 正四面体の1辺の長さを定義する。
    a = 125
    b = 93  # a > b　固定値

    t1 = 1 # 壁面内側の厚さ
    t2 = 6 # 底面内側の厚さ

    hankei = (a / 2.0) / np.sin(np.radians(36))
    mw = 4 * hankei ** 2 * np.sin(np.radians(18)) ** 2
    takasa = np.sqrt(a**2 - mw)

    # 切頂20面体の一番上の点群の一つの点
    tamentai[0, 0, 1, 0, 0] = (a / 2.0) / np.tan(np.radians(36))
    tamentai[0, 0, 1, 0, 1] = a / 2.0
    tamentai[0, 0, 1, 0, 2] = takasa / 2.0

    ma = np.degrees(np.arccos(1/(np.sqrt(3) * np.tan(np.radians(36)))))

    # 正20面体の一番上の頂点
    tamentai[0, 0, 0, 0, 0] = 0
    tamentai[0, 0, 0, 0, 1] = 0
    tamentai[0, 0, 0, 0, 2] = tamentai[0, 0, 1, 0, 2] + np.sqrt(3*(a**2)/4.0 - tamentai[0, 0, 1, 0, 0]**2)

    # 72度回転で上側五角形を形成 内側の五角形も含む
    angle = np.radians(72)
    for v in [1]:
        for i in range(1, 5):
            tamentai[0, 0, v, i, 0] = (tamentai[0, 0, v, i-1, 0] * np.cos(angle) - tamentai[0, 0, v, i-1, 1] * np.sin(angle))
            tamentai[0, 0, v, i, 1] = (tamentai[0, 0, v, i-1, 0] * np.sin(angle) + tamentai[0, 0, v, i-1, 1] * np.cos(angle))
            tamentai[0, 0, v, i, 2] = tamentai[0, 0, v, 0, 2]

    # 頂点(0,0,0,0,:) と上側五角形(0,0,1,i,:) を 1:2 の比で内分 -> tamentai[0,1,0,i,:]
    # さらに相似比 b/a で縮小したもの -> tamentai[1,1,0,i,:]
    for i in range(0, 5):
        pty1 = tamentai[0, 0, 0, 0, :]
        pty2 = tamentai[0, 0, 1, i, :]
        point_sets1 = [pty1, pty2]

        x1, y1, z1 = santobun(point_sets1)
 
        tamentai[0, 1, 0, i, :] = [x1, y1, z1]
        tamentai[1, 1, 0, i, :] = [b*x1/a, b*y1/a, b*z1/a]

    tamentai[0, 0, 2, 0, 0] = tamentai[0, 1, 0, 0, 0] - t1
    tamentai[0, 0, 2, 0, 1] = tamentai[0, 1, 0, 0, 1] - t1 * np.tan(np.radians(36))
    tamentai[0, 0, 2, 0, 2] = tamentai[0, 1, 0, 0, 2]

    tamentai[1, 1, 2, 0, 0] = tamentai[0, 0, 2, 0, 0] * (tamentai[1, 1, 0, 0, 2] + t2) / tamentai[0, 0, 2, 0, 2]
    tamentai[1, 1, 2, 0, 1] = tamentai[0, 0, 2, 0, 1] * (tamentai[1, 1, 0, 0, 2] + t2) / tamentai[0, 0, 2, 0, 2]
    tamentai[1, 1, 2, 0, 2] = tamentai[1, 1, 0, 0, 2] + t2

    for i in range(1, 5):
        tamentai[0, 0, 2, i, 0] = tamentai[0, 0, 2, i-1, 0] * np.cos(angle) - tamentai[0, 0, 2, i-1, 1] * np.sin(angle)
        tamentai[0, 0, 2, i, 1] = tamentai[0, 0, 2, i-1, 0] * np.sin(angle) + tamentai[0, 0, 2, i-1, 1] * np.cos(angle)
        tamentai[0, 0, 2, i, 2] = tamentai[0, 0, 2, 0, 2]

        tamentai[1, 1, 2, i, 0] = tamentai[1, 1, 2, i-1, 0] * np.cos(angle) - tamentai[1, 1, 2, i-1, 1] * np.sin(angle)
        tamentai[1, 1, 2, i, 1] = tamentai[1, 1, 2, i-1, 0] * np.sin(angle) + tamentai[1, 1, 2, i-1, 1] * np.cos(angle)
        tamentai[1, 1, 2, i, 2] = tamentai[1, 1, 2, 0, 2]

    # --- ここから hesa の計算（1枚の三角形を 3等分する例） ---
    dfg1 = tamentai[0, 0, 0, 0, :]
    dfg2 = tamentai[0, 0, 1, 0, :]
    dfg3 = tamentai[0, 0, 1, 4, :]

    x4, y4, z4, x5, y5, z5 = santobun2([dfg1, dfg3])
    x2, y2, z2, x3, y3, z3 = santobun2([dfg2, dfg1])

    theta_rad = np.atan2(y5+y2, x5+x2)
    kei = np.degrees(theta_rad)

    kei = (90-kei)

    # dfg2-dfg3 の 3等分点2つ
    x1, y1, z1, x0, y0, z0 = santobun2([dfg2, dfg3])

    x11, y11, z11 = b * x1 / a, b * y1 / a, b * z1 / a
    x1, y1, z1 = y_θ(x1, y1, z1, kei)

    hesa[0, 0, 1, :] = [x1, y1, z1]

    x01, y01, z01 = b * x0 / a, b * y0 / a, b * z0 / a
    x0, y0, z0 = y_θ(x0, y0, z0, kei)

    hesa[0, 0, 0, :] = [x0, y0, z0]

    x11, y11, z11 = y_θ(x11, y11, z11, kei)

    hesa[0, 1, 1, :] = [x11, y11, z11]

    x01, y01, z01 = y_θ(x01, y01, z01, kei)

    hesa[0, 1, 0, :] = [x01, y01, z01]

    # dfg1-dfg2 の 3等分点2つ
    x2, y2, z2, x3, y3, z3 = santobun2([dfg2, dfg1])

    x21, y21, z21 =  b * x2 / a, b * y2 / a, b * z2 / a
    x2, y2, z2 = y_θ(x2, y2, z2, kei)

    hesa[0, 0, 2, :] = [x2, y2, z2]

    x31, y31, z31 =  b * x3 / a, b * y3 / a, b * z3 / a
    x3, y3, z3 = y_θ(x3, y3,z3, kei)

    hesa[0, 0, 3, :] = [x3, y3, z3]

    x21, y21, z21 = y_θ(x21, y21, z21, kei)

    hesa[0, 1, 2, :] = [x21, y21, z21]

    x31, y31, z31 = y_θ(x31, y31, z31, kei)

    hesa[0, 1, 3, :] = [x31, y31, z31]

    # dfg3-dfg1 の 3等分点2つ
    x4, y4, z4, x5, y5, z5 = santobun2([dfg1, dfg3])
 
    x41, y41, z41 =  b * x4 / a, b * y4 / a, b * z4 / a
    x4, y4, z4 = y_θ(x4, y4, z4, kei)

    hesa[0, 0, 4, :] = [x4, y4, z4]

    x51, y51, z51 =  b * x5 / a, b * y5 / a, b * z5 / a
    x5, y5, z5 = y_θ(x5, y5,z5, kei)

    hesa[0, 0, 5, :] = [x5, y5, z5]

    x41, y41, z41 = y_θ(x41, y41, z41, kei)

    hesa[0, 1, 4, :] = [x41, y41, z41]

    x51, y51, z51 = y_θ(x51, y51, z51, kei)

    hesa[0, 1, 5, :] = [x51, y51, z51]

    # 重心 回転前
    jushin_x = (x5+x2) / 2.0
    jushin_y = (y5+y2) / 2.0
    jushin_z = (z5+z2) / 2.0

    nagasa = np.sqrt((x5-x2)**2+(y5-y2)**2+(z5-z2)**2) / 2.0

    kyu1 = np.sqrt(jushin_x**2+jushin_y**2+jushin_z**2)
    kyu2 = b * kyu1 / a

    lo = (nagasa - t1) / nagasa

    link = np.zeros((2, 6, 3), dtype=float)

    lstx = [x0, x1, x2, x3, x4, x5]
    lsty = [y0, y1, y2, y3, y4, y5]
    lstz = [z0, z1, z2, z3, z4, z5]
    for i in range(0, 6):
        x = lstx[i]
        y = lsty[i]
        z = lstz[i]

        link[0, i, 0] = (1-lo) * jushin_x + lo * x
        link[0, i, 1] = (1-lo) * jushin_y + lo * y
        link[0, i, 2] = (1-lo) * jushin_z + lo * z

        link[1, i, 0] = (kyu2 + t2) * link[0, i, 0] / kyu1
        link[1, i, 1] = (kyu2 + t2) * link[0, i, 1] / kyu1
        link[1, i, 2] = (kyu2 + t2) * link[0, i, 2] / kyu1

    for v in range(0, 2):
        for i in range(0, 6):
            hesa[1, v, i, 0] = link[v, i, 0] * np.cos(np.radians(kei)) + link[v, i, 2] * np.sin(np.radians(kei))
            hesa[1, v, i, 1] = link[v, i, 1]
            hesa[1, v, i, 2] = -link[v, i, 0] * np.sin(np.radians(kei)) + link[v, i, 2] * np.cos(np.radians(kei))

    # --- tamentai で作ったリングの STL ---
    triangle_index_list_tamentai = []

    # 例: 内側リングの 3点で1枚の三角形（お好みで変更）
    triangle_index_list_tamentai.append(((1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2)))
    triangle_index_list_tamentai.append(((1, 1, 0, 0), (1, 1, 0, 2), (1, 1, 0, 3)))
    triangle_index_list_tamentai.append(((1, 1, 0, 0), (1, 1, 0, 3), (1, 1, 0, 4)))

    for i in range(0, 5):
        triangle_index_list_tamentai.append(((0, 1, 0, i), (0, 1, 0, (i+1) % 5), (1, 1, 0, i)))
        triangle_index_list_tamentai.append(((1, 1, 0, i), (0, 1, 0, (i+1) % 5), (1, 1, 0, (i+1) % 5)))
    
    for i in range(0, 3):
        triangle_index_list_tamentai.append(((1, 1, 2, 0), (1, 1, 2, i+1), (1, 1, 2, i+2)))

    for i in range(0, 5):
        triangle_index_list_tamentai.append(((1, 1, 2, i), (0, 0, 2, i), (0, 0, 2, (i+1)%5)))
        triangle_index_list_tamentai.append(((1, 1, 2, i), (1, 1, 2, (i+1)%5), (0, 0, 2, (i+1)%5)))
        triangle_index_list_tamentai.append(((0, 0, 2, i), (0, 1, 0, i), (0, 1, 0, (i+1)%5)))
        triangle_index_list_tamentai.append(((0, 0, 2, i), (0, 0, 2, (i+1)%5), (0, 1, 0, (i+1)%5)))

    export_stl_from_triangle_indices(triangle_index_list_tamentai, filename="test_inner_ring_tamentai.stl",)

    # --- hesa で三角形メッシュを作って STL 出力 ---
    triangle_index_list_hesa = []

    # ここでは 6角形リング2本をつなぐチューブ状メッシュのイメージ
    for i in range(0, 4):
        triangle_index_list_hesa.append(((0, 1, 0), (0, 1, i+1), (0, 1, i+2)))
      #  triangle_index_list_hesa.append(((1, 1, 0), (1, 1, i+1), (1, 1, i+2)))

    #for i in range(0, 6):
     #   triangle_index_list_hesa.append(((0, 0, i), (0, 0, (i+1)%6), (0, 1, i)))
    #    triangle_index_list_hesa.append(((0, 0, (i+1)%6), (0, 1, (i+1)%6), (0, 1, i)))

    export_stl_from_hesa_triangle_indices(triangle_index_list_hesa, filename="hesa_ring.stl",)

if __name__ == "__main__":
    main()
