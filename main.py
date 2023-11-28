import math
import matplotlib.pyplot as plot
import random


def gen_20_points():
    points = []

    for i in range(20):
        points.append((random.randint(-5000, 5000), random.randint(-5000, 5000)))

    return points


def gen_other_40000_points(initial):
    for i in range(40000):
        (x, y) = initial[random.randint(0, len(initial) - 1)]
        (x_offset_min, x_offset_max, y_offset_min, y_offset_max) = (
            -min(100, x + 5000),
            min(100, 5000 - x),
            -min(100, y + 5000),
            min(100, 5000 - y)
        )

        x_offset = random.randint(x_offset_min, x_offset_max)
        y_offset = random.randint(y_offset_min, y_offset_max)

        initial.append((x + x_offset, y + y_offset))


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def calculate_average_euclidean_distance(cluster):
    total_distance = 0
    num_pairs = 0

    for i, a in enumerate(cluster):
        for b in cluster[i+1:]:
            total_distance += euclidean_distance(a, b)
            num_pairs += 1

    return total_distance / num_pairs if num_pairs > 0 else 0


def k_means_centroid_impl(points, k, max_iterations=1000):
    centroids = random.sample(points, k)
    clusters = [[] for _ in range(k)]

    for _ in range(max_iterations):
        new_clusters = [[] for _ in range(k)]

        for point in points:
            closest_centroid_idx = min(range(k), key=lambda i: euclidean_distance(point, centroids[i]))
            new_clusters[closest_centroid_idx].append(point)

        if clusters == new_clusters:
            break

        clusters = new_clusters

        for j in range(k):
            if clusters[j]:
                c_x = sum(point[0] for point in clusters[j]) / len(clusters[j])
                c_y = sum(point[1] for point in clusters[j]) / len(clusters[j])

                centroids[j] = (c_x, c_y)

    return clusters, centroids


def k_means_centroid(points, k):
    while True:
        print(f"Attempting: K = {k}")

        clusters_temp, centroids_temp = k_means_centroid_impl(points, k)

        for cluster in clusters_temp:
            if calculate_average_euclidean_distance(cluster) > 500:
                k += 1

                break
        else:
            print(f"Found: K = {k}")

            clusters, centroids = (clusters_temp, centroids_temp)

            break

    plot.figure(figsize=(20, 20))
    plot.xlim(-5000, 5000)
    plot.ylim(-5000, 5000)
    plot.xlabel('X-axis')
    plot.ylabel('Y-axis')

    for i, cluster in enumerate(clusters):
        x_coords = [t[0] for t in cluster]
        y_coords = [t[1] for t in cluster]

        c_x, c_y = centroids[i]

        plot.scatter(x_coords, y_coords, label=f'Cluster {i+1}')
        plot.scatter(c_x, c_y, color='black')

    plot.savefig('output_plot.png', dpi=300, bbox_inches='tight')
    plot.show()


if __name__ == "__main__":
    data = gen_20_points()
    gen_other_40000_points(data)

    k_means_centroid(data, 15)
