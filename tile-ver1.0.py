import numpy as np
import matplotlib.pyplot as plt
import trimesh

# tamentai: (2, 2, 2, 5, 3)
tamentai = np.zeros((2, 2, 2, 5, 3), dtype=float)
# hesa: (2, 6, 3)  2リング × 各6点 × xyz
hesa = np.zeros((2, 6, 3), dtype=float)

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

def build_vertices_and_faces_from_hesa(triangle_index_list):
    """
    triangle_index_list:
        [ ((ring0, idx0), (ring1, idx1), (ring2, idx2)), ... ]

    hesa[ring, idx, :] を参照して、
      - vertices: (N,3)
      - faces   : (M,3)
    を返す。
    """
    vertices = []
    index_to_vid = {}  # (ring, idx) -> vertex id
    faces = []

    for tri in triangle_index_list:
        face_vids = []
        for idx2 in tri:
            if idx2 not in index_to_vid:
                x, y, z = hesa[idx2[0], idx2[1], :]
                vid = len(vertices)
                vertices.append([x, y, z])
                index_to_vid[idx2] = vid
            face_vids.append(index_to_vid[idx2])
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

# --- メイン処理 ---

def main():
    global tamentai, hesa

    # 正四面体の1辺の長さを定義する。
    a = 220
    b = 200  # a > b

    hankei = (a / 2.0) / np.sin(np.radians(36))
    mw = 4 * hankei ** 2 * np.sin(np.radians(18)) ** 2
    takasa = np.sqrt(a**2 - mw)

    # 正二十面体の「上の五角形頂点」と「頂点」の例
    tamentai[0, 0, 1, 0, 0] = (a / 2.0) / np.tan(np.radians(36))
    tamentai[0, 0, 1, 0, 1] = a / 2.0
    tamentai[0, 0, 1, 0, 2] = takasa / 2.0

    tamentai[0, 0, 0, 0, 0] = 0
    tamentai[0, 0, 0, 0, 1] = 0
    tamentai[0, 0, 0, 0, 2] = tamentai[0, 0, 1, 0, 2] + np.sqrt(
        3*(a**2)/4.0 - tamentai[0, 0, 1, 0, 0]**2
    )

    # 72度回転で上側五角形を生成
    angle = np.radians(72)
    for i in range(1, 5):
        tamentai[0, 0, 1, i, 0] = (
            tamentai[0, 0, 1, i-1, 0] * np.cos(angle)
            - tamentai[0, 0, 1, i-1, 1] * np.sin(angle)
        )
        tamentai[0, 0, 1, i, 1] = (
            tamentai[0, 0, 1, i-1, 0] * np.sin(angle)
            + tamentai[0, 0, 1, i-1, 1] * np.cos(angle)
        )
        tamentai[0, 0, 1, i, 2] = tamentai[0, 0, 1, 0, 2]

    # 頂点(0,0,0,0,:) と上側五角形(0,0,1,i,:) を 1:2 の比で内分 -> tamentai[0,1,0,i,:]
    # さらに相似比 b/a で縮小したもの -> tamentai[1,1,0,i,:]
    for i in range(0, 5):
        pty1 = tamentai[0, 0, 0, 0, :]
        pty2 = tamentai[0, 0, 1, i, :]
        point_sets1 = [pty1, pty2]

        x1, y1, z1 = santobun(point_sets1)

        tamentai[0, 1, 0, i, :] = [x1, y1, z1]
        tamentai[1, 1, 0, i, :] = [b*x1/a, b*y1/a, b*z1/a]

    # --- ここから hesa の計算（1枚の三角形を 3等分する例） ---
    dfg1 = tamentai[0, 0, 0, 0, :]
    dfg2 = tamentai[0, 0, 1, 0, :]
    dfg3 = tamentai[0, 0, 1, 1, :]

    # dfg1-dfg2 の 3等分点2つ
    x2, y2, z2, x3, y3, z3 = santobun2([dfg1, dfg2])
    hesa[0, 0, :] = [x2, y2, z2]
    hesa[0, 1, :] = [x3, y3, z3]

    # dfg2-dfg3 の 3等分点2つ
    x4, y4, z4, x5, y5, z5 = santobun2([dfg2, dfg3])
    hesa[0, 2, :] = [x4, y4, z4]
    hesa[0, 3, :] = [x5, y5, z5]

    # dfg3-dfg1 の 3等分点2つ
    x6, y6, z6, x7, y7, z7 = santobun2([dfg3, dfg1])
    hesa[0, 4, :] = [x6, y6, z6]
    hesa[0, 5, :] = [x7, y7, z7]

    # 縮小版 (相似比 b/a)
    for i in range(0, 6):
        for j in range(0, 3):
            hesa[1, i, j] = b * hesa[0, i, j] / a

    # --- tamentai で作ったリングの STL ---
    triangle_index_list_tamentai = []

    # 例: 内側リングの 3点で1枚の三角形（お好みで変更）
    triangle_index_list_tamentai.append(
        ((1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2))
    )
    triangle_index_list_tamentai.append(
        ((1, 1, 0, 0), (1, 1, 0, 2), (1, 1, 0, 3))
    )
    triangle_index_list_tamentai.append(
        ((1, 1, 0, 0), (1, 1, 0, 3), (1, 1, 0, 4))
    )

    for i in range(0, 5):
        triangle_index_list_tamentai.append(
            ((0, 1, 0, i), (0, 1, 0, (i+1) % 5), (1, 1, 0, i))
        )
        triangle_index_list_tamentai.append(
            ((1, 1, 0, i), (0, 1, 0, (i+1) % 5), (1, 1, 0, (i+1) % 5))
        )

    export_stl_from_triangle_indices(
        triangle_index_list_tamentai,
        filename="test_inner_ring_tamentai.stl",
    )

    # --- hesa で三角形メッシュを作って STL 出力 ---
    triangle_index_list_hesa = []

    # ここでは 6角形リング2本をつなぐチューブ状メッシュのイメージ
    for i in range(0, 4):
        triangle_index_list_hesa.append(((1, 0), (1, i+1), (1, i+2)))

    for i in range(0, 6):
        triangle_index_list_hesa.append(((0, i), (0, (i+1)%6), (1, i)))
        triangle_index_list_hesa.append(((0, (i+1)%6), (1, (i+1)%6), (1, i)))

    export_stl_from_hesa_triangle_indices(
        triangle_index_list_hesa,
        filename="hesa_ring.stl",
    )

    # 可視化
    show_points()

def show_points():
    """
    tamentai[0,1,0,i,:] と tamentai[1,1,0,i,:] の点を
    print と 3D グラフで表示
    """
    pts_outer = tamentai[0, 1, 0, :, :]  # shape = (5, 3)
    pts_inner = tamentai[1, 1, 0, :, :]  # shape = (5, 3)

    print("tamentai[0, 1, 0, i, :] の点座標:")
    print(pts_outer)
    print()
    print("tamentai[1, 1, 0, i, :] の点座標:")
    print(pts_inner)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts_outer[:, 0], pts_outer[:, 1], pts_outer[:, 2],
               marker='o', label='tamentai[0,1,0,i,:]')

    ax.scatter(pts_inner[:, 0], pts_inner[:, 1], pts_inner[:, 2],
               marker='^', label='tamentai[1,1,0,i,:]')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Points in tamentai[0,1,0,i,:] and tamentai[1,1,0,i,:]')

    plt.show()


if __name__ == "__main__":
    main()
