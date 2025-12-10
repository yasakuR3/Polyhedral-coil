import numpy as np
import matplotlib.pyplot as plt
import trimesh

def set_equal_aspect_3d(ax, X, Y, Z):
    # 3Dで見た目のスケールを揃える補助関数
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    z_min, z_max = np.min(Z), np.max(Z)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def santobun(point_set):
    p1 = point_set[0]
    p2 = point_set[1]

    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]

    x2 = p2[0]
    y2 = p2[1]
    z2 = p2[2]

    return (2*x1+x2)/3.0, (2*y1+y2)/3.0, (2*z1+z2)/3.0, (x1+2*x2)/3.0, (y1+2*y2)/3.0, (z1+2*z2)/3.0

def build_mesh_from_triangles(tamentai, triangles):

    vertices = []
    faces = []
    vertex_map = {}

    for tri in triangles:
        face = []
        for key in tri:
            if key not in vertex_map:
                k, f, i = key
                x, y, z = tamentai[k, f, i, :]
                vertex_map[key] = len(vertices)
                vertices.append([x, y, z])
            face.append(vertex_map[key])
        faces.append(face)

    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh

def main():
    # ここで配列を準備（あなたの添
    tamentai = np.zeros((2, 8, 10, 3), dtype=float)

    # 正二十面体の1辺の長さを指定する。指定すうと一意的に切頂20面体が決定する。
    a = 200 # 内側の切頂20面体
    # b = 300 (b > a) # 外側の切頂20面体

    hankei = (a / 2.0) / np.sin(np.radians(36))
    we1 = a ** 2 - 4 * hankei ** 2 * np.sin(np.radians(18)) ** 2

    tamentai[0, 0, 0, 0] = (a / 2.0) / np.tan(np.radians(36))
    tamentai[0, 0, 0, 1] = a / 2.0
    tamentai[0, 0, 0, 2] = 0 

    angle = np.radians(72)

    for i in range(1, 5):
        tamentai[0, 0, i, 0] = tamentai[0, 0, i-1, 0] * np.cos(angle) - tamentai[0, 0, i-1, 1] * np.sin(angle)
        tamentai[0, 0, i, 1] = tamentai[0, 0, i-1, 0] * np.sin(angle) + tamentai[0, 0, i-1, 1] * np.cos(angle)
        tamentai[0, 0, i, 2] = tamentai[0, 0, 0, 2]

    tamentai[0, 1, 0, 0] = 0
    tamentai[0, 1, 0, 1] = 0
    tamentai[0, 1, 0, 2] = np.sqrt((3 / 4) * a ** 2 - tamentai[0, 0, 0, 0] ** 2)

    angle2 = np.radians(36)

    tamentai[0, 2, 0, 0] = tamentai[0, 0, 4, 0] * np.cos(angle2) - tamentai[0, 0, 4, 1] * np.sin(angle2)
    tamentai[0, 2, 0, 1] = 0 
    tamentai[0, 2, 0, 2] = -np.sqrt(we1)

    for i in range(1, 5):
        tamentai[0, 2, i, 0] = tamentai[0, 2, i-1, 0] * np.cos(angle) - tamentai[0, 2, i-1, 1] * np.sin(angle)
        tamentai[0, 2, i, 1] = tamentai[0, 2, i-1, 0] * np.sin(angle) + tamentai[0, 2, i-1, 1] * np.cos(angle)
        tamentai[0, 2, i, 2] = tamentai[0, 2, 0, 2]

    tamentai[0, 3, 0, 0] = 0
    tamentai[0, 3, 0, 1] = 0
    tamentai[0, 3, 0, 2] = tamentai[0, 2, 0, 2] - tamentai[0, 1, 0, 2]

    for i in range(0, 5):
        pty1 = tamentai[0, 1, 0, :]
        pty2 = tamentai[0, 0, i, :]

        point_sets1 = [pty1, pty2]

        x1, y1, z1, x2, y2, z2 = santobun(point_sets1)

        tamentai[1, 0, i, :] = [x1, y1, z1]
        tamentai[1, 1, i, :] = [x2, y2, z2]

    n = 5
    j = 0
    for i in range(0, 5):
        hgh1 = tamentai[0, 0, i, :]
        hgh2 = tamentai[0, 0, (i+1)%n, :]

        point_sets2 = [hgh1, hgh2]

        x3, y3, z3, x4, y4, z4 = santobun(point_sets2)

        tamentai[1, 2, j, :] = [x3, y3, z3]
        tamentai[1, 2, j+1, :] = [x4, y4, z4]

        j = j + 2

    k = 0
    for i in range(0, 5):
        asd1 = tamentai[0, 0, i, :]
        asd2 = tamentai[0, 2, i, :]

        point_sets3 = [asd1, asd2]

        x5, y5, z5, x6, y6, z6 = santobun(point_sets3)

        tamentai[1, 3, k, :] = [x5, y5, z5]
        tamentai[1, 4, k, :] = [x6, y6, z6]

        poi1 = tamentai[0, 0, i, :]
        poi2 = tamentai[0, 2, (i+1)%n, :]

        point_sets4 = [poi1, poi2]

        x7, y7, z7, x8, y8, z8 = santobun(point_sets4)

        tamentai[1, 3, k+1, :] = x7, y7, z7
        tamentai[1, 4, k+1, :] = x8, y8, z8

        k = k + 2
    
    h = 0
    for i in range(0, 5):
        qwe1 = tamentai[0, 2, i, :]
        qwe2 = tamentai[0, 2, (i+1)%n, :]

        point_sets5 = [qwe1, qwe2]

        x9, y9, z9, x10, y10, z10 = santobun(point_sets5)

        tamentai[1, 5, h, :] = [x9, y9, z9]
        tamentai[1, 5, h+1, :] = [x10, y10, z10]

        h = h + 2

    for i in range(0, 5):
        zse1 = tamentai[0, 2, i, :]
        zse2 = tamentai[0, 3, 0, :]

        point_sets6 = [zse1, zse2]

        x11, y11, z11, x12, y12, z12 = santobun(point_sets6)

        tamentai[1, 6, i, :] = [x11, y11, z11]
        tamentai[1, 7, i, :] = [x12, y12, z12]
    
    triangles = []

    # 一番上と一番下の面
    for i in range(0, 3):
        triangles.append(((1, 0, 0), (1, 0, i+1), (1, 0, i+2)))
        triangles.append(((1, 7, 0), (1, 7, i+1), (1, 7, i+2)))
    
    # 一番上から二番目の面群
    k=0
    m=1
    for i in range(0, 5):
        triangles.append(((1, 0, i), (1, 0, (i+1)%n), (1, 1, i)))
        triangles.append(((1, 0, (i+1)%n), (1, 1, (i+1)%n), (1, 1, i)))
        triangles.append(((1, 1, i), (1, 2, k), (1, 1, (i+1)%n)))
        triangles.append(((1, 2, k), (1, 2, m), (1, 1, (i+1)%n)))

        k = k + 2
        m = m + 2
    
    # 一番上から3番目の面群
    l = 1
    t = 0
    for i in range(0, 5):
        triangles.append(((1, 1, (i+1)%n), (1, 2, l), (1, 2, (l+1)%10)))
        triangles.append(((1, 2, l), (1, 2, (l+1)%10), (1, 3, (l+1)%10)))
        triangles.append(((1, 3, (l+1)%10), (1, 2, (l+1)%10), (1, 3, (l+2)%10)))

        l = l + 2
        t = t + 2

    # 一番上から4番目の面群
    triangles.append(((1, 2, 1), (1, 2, 0), (1, 3, 1)))
    triangles.append(((1, 2, 3), (1, 2, 2), (1, 3, 3)))
    triangles.append(((1, 2, 5), (1, 2, 4), (1, 3, 5)))
    triangles.append(((1, 2, 7), (1, 2, 6), (1, 3, 7)))
    triangles.append(((1, 2, 9), (1, 2, 8), (1, 3, 9)))

    triangles.append(((1, 2, 1), (1, 3, 2), (1, 3, 1)))
    triangles.append(((1, 2, 3), (1, 3, 4), (1, 3, 3)))
    triangles.append(((1, 2, 5), (1, 3, 6), (1, 3, 5)))
    triangles.append(((1, 2, 7), (1, 3, 8), (1, 3, 7)))
    triangles.append(((1, 2, 9), (1, 3, 0), (1, 3, 9)))

    triangles.append(((1, 4, 1), (1, 3, 2), (1, 3, 1)))
    triangles.append(((1, 4, 3), (1, 3, 4), (1, 3, 3)))
    triangles.append(((1, 4, 5), (1, 3, 6), (1, 3, 5)))
    triangles.append(((1, 4, 7), (1, 3, 8), (1, 3, 7)))
    triangles.append(((1, 4, 9), (1, 3, 0), (1, 3, 9)))

    triangles.append(((1, 4, 1), (1, 3, 2), (1, 4, 2)))
    triangles.append(((1, 4, 3), (1, 3, 4), (1, 4, 4)))
    triangles.append(((1, 4, 5), (1, 3, 6), (1, 4, 6)))
    triangles.append(((1, 4, 7), (1, 3, 8), (1, 4, 8)))
    triangles.append(((1, 4, 9), (1, 3, 0), (1, 4, 0)))

    # 一番上から5番目の面群
    triangles.append(((1, 3, 3), (1, 3, 2), (1, 4, 2)))
    triangles.append(((1, 3, 5), (1, 3, 4), (1, 4, 4)))
    triangles.append(((1, 3, 7), (1, 3, 6), (1, 4, 6)))
    triangles.append(((1, 3, 9), (1, 3, 8), (1, 4, 8)))
    triangles.append(((1, 3, 1), (1, 3, 0), (1, 4, 0)))

    triangles.append(((1, 3, 3), (1, 4, 3), (1, 4, 2)))
    triangles.append(((1, 3, 5), (1, 4, 5), (1, 4, 4)))
    triangles.append(((1, 3, 7), (1, 4, 7), (1, 4, 6)))
    triangles.append(((1, 3, 9), (1, 4, 9), (1, 4, 8)))
    triangles.append(((1, 3, 1), (1, 4, 1), (1, 4, 0)))

    triangles.append(((1, 5, 3), (1, 4, 3), (1, 4, 2)))
    triangles.append(((1, 5, 5), (1, 4, 5), (1, 4, 4)))
    triangles.append(((1, 5, 7), (1, 4, 7), (1, 4, 6)))
    triangles.append(((1, 5, 9), (1, 4, 9), (1, 4, 8)))
    triangles.append(((1, 5, 1), (1, 4, 1), (1, 4, 0)))

    triangles.append(((1, 5, 3), (1, 5, 2), (1, 4, 2)))
    triangles.append(((1, 5, 5), (1, 5, 4), (1, 4, 4)))
    triangles.append(((1, 5, 7), (1, 5, 6), (1, 4, 6)))
    triangles.append(((1, 5, 9), (1, 5, 8), (1, 4, 8)))
    triangles.append(((1, 5, 1), (1, 5, 0), (1, 4, 0)))

    # 一番上から6番目の面群
    triangles.append(((1, 5, 1), (1, 4, 1), (1, 4, 2)))
    triangles.append(((1, 5, 3), (1, 4, 3), (1, 4, 4)))
    triangles.append(((1, 5, 5), (1, 4, 5), (1, 4, 6)))
    triangles.append(((1, 5, 7), (1, 4, 7), (1, 4, 8)))
    triangles.append(((1, 5, 9), (1, 4, 9), (1, 4, 0)))

    triangles.append(((1, 5, 1), (1, 5, 2), (1, 4, 2)))
    triangles.append(((1, 5, 3), (1, 5, 4), (1, 4, 4)))
    triangles.append(((1, 5, 5), (1, 5, 6), (1, 4, 6)))
    triangles.append(((1, 5, 7), (1, 5, 8), (1, 4, 8)))
    triangles.append(((1, 5, 9), (1, 5, 0), (1, 4, 0)))

    triangles.append(((1, 5, 1), (1, 5, 2), (1, 6, 1)))
    triangles.append(((1, 5, 3), (1, 5, 4), (1, 6, 2)))
    triangles.append(((1, 5, 5), (1, 5, 6), (1, 6, 3)))
    triangles.append(((1, 5, 7), (1, 5, 8), (1, 6, 4)))
    triangles.append(((1, 5, 9), (1, 5, 0), (1, 6, 0)))

    # 一番上から7番目の面群
    triangles.append(((1, 5, 1), (1, 5, 0), (1, 6, 0)))
    triangles.append(((1, 5, 3), (1, 5, 2), (1, 6, 1)))
    triangles.append(((1, 5, 5), (1, 5, 4), (1, 6, 2)))
    triangles.append(((1, 5, 7), (1, 5, 6), (1, 6, 3)))
    triangles.append(((1, 5, 9), (1, 5, 8), (1, 6, 4)))

    triangles.append(((1, 5, 1), (1, 6, 1), (1, 6, 0)))
    triangles.append(((1, 5, 3), (1, 6, 2), (1, 6, 1)))
    triangles.append(((1, 5, 5), (1, 6, 3), (1, 6, 2)))
    triangles.append(((1, 5, 7), (1, 6, 4), (1, 6, 3)))
    triangles.append(((1, 5, 9), (1, 6, 0), (1, 6, 4)))

    triangles.append(((1, 7, 1), (1, 6, 1), (1, 6, 0)))
    triangles.append(((1, 7, 2), (1, 6, 2), (1, 6, 1)))
    triangles.append(((1, 7, 3), (1, 6, 3), (1, 6, 2)))
    triangles.append(((1, 7, 4), (1, 6, 4), (1, 6, 3)))
    triangles.append(((1, 7, 0), (1, 6, 0), (1, 6, 4)))

    triangles.append(((1, 7, 1), (1, 7, 0), (1, 6, 0)))
    triangles.append(((1, 7, 2), (1, 7, 1), (1, 6, 1)))
    triangles.append(((1, 7, 3), (1, 7, 2), (1, 6, 2)))
    triangles.append(((1, 7, 4), (1, 7, 3), (1, 6, 3)))
    triangles.append(((1, 7, 0), (1, 7, 4), (1, 6, 4)))


    mesh = build_mesh_from_triangles(tamentai, triangles)
    mesh.export("tamentai_shell.stl")  # カレントディレクトリに出力

    # ===== ここから「点の表示」 =====
    pts = tamentai[1].reshape(-1, 3)  # ← これで tamentai[1, :, :, :] だけに限定

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=40)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("tamentai[1] points only")

    set_equal_aspect_3d(ax, pts[:, 0], pts[:, 1], pts[:, 2])
    plt.show()

if __name__ == "__main__":
    main()
