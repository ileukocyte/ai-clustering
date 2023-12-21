import math
import time

import matplotlib.pyplot as plot
import random


### Dataset generation
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


### Auxiliary functions
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def calculate_avg_distance_from_center(cluster, center):
    return sum(euclidean_distance(center, p) for p in cluster) / len(cluster)


def calculate_centroid(cluster):
    if not cluster:
        return None

    c_x = sum(point[0] for point in cluster) / len(cluster)
    c_y = sum(point[1] for point in cluster) / len(cluster)

    return c_x, c_y


### Algorithm implementations
def k_means(points, k, max_iterations=1000):
    start_time = time.time()

    # Initial random centroid generation
    centroids = random.sample(points, k)
    clusters = [[] for _ in range(k)]

    for _ in range(max_iterations):
        # Generate K clusters
        new_clusters = [[] for _ in range(k)]

        # Assign each point to the nearest centroid (based on the Euclidean distance)
        for point in points:
            closest_centroid_index = min(range(k), key=lambda i: euclidean_distance(point, centroids[i]))
            new_clusters[closest_centroid_index].append(point)

        # Check for convergence by comparing old clusters with new clusters
        if clusters == new_clusters:
            break

        clusters = new_clusters

        # Update centroids to the mean of the points within each cluster
        for j in range(k):
            if clusters[j]:
                centroids[j] = calculate_centroid(clusters[j])

    return clusters, centroids, time.time() - start_time


def k_medoids(points, k, max_iterations=1000):
    start_time = time.time()

    # Initial random medoid selection
    medoids = random.sample(points, k)
    clusters = [[] for _ in range(k)]

    for _ in range(max_iterations):
        # Generate K clusters
        new_clusters = [[] for _ in range(k)]

        # Assign each point to the nearest medoid (based on the Euclidean distance)
        for point in points:
            closest_medoid_index = min(range(k), key=lambda i: euclidean_distance(point, medoids[i]))
            new_clusters[closest_medoid_index].append(point)

        # Check for convergence by comparing old clusters with new clusters
        if clusters == new_clusters:
            break

        clusters = new_clusters

        # Update medoids by choosing the point that minimizes total distance within clusters
        for j in range(k):
            if clusters[j]:
                # Calculate distances between points within the cluster
                distances = [sum(euclidean_distance(point, p) for p in clusters[j]) for point in clusters[j]]
                # Find the index of the point that minimizes the total distance
                min_distance_index = distances.index(min(distances))
                medoids[j] = clusters[j][min_distance_index]

    return clusters, medoids, time.time() - start_time


def divisive_clustering(points):
    start_time = time.time()

    # A single cluster in the beginning
    clusters = [points]

    successful = False

    while len(clusters) < len(points) and not successful:
        max_cluster = max(clusters, key=len)  # Find the cluster with the maximum number of points

        if max_cluster:
            # Using K-means to halve the largest cluster
            split, _, _ = k_means(max_cluster, 2)

            # Replace the largest cluster with the two subclusters
            clusters.remove(max_cluster)
            clusters.extend(split)

            successful = not any(calculate_avg_distance_from_center(cl, calculate_centroid(cl)) > 500 for cl in clusters)
        else:
            break

    return clusters, successful, time.time() - start_time


### Algorithm visualizations
def k_means_visualize(points, k):
    print(f'Attempting K-means (K = {k})...')

    clusters, centroids, time_secs = k_means(points, k)

    print(f"Execution time: {time_secs}s")

    successful = not any(calculate_avg_distance_from_center(cl, centroids[i]) > 500 for i, cl in enumerate(clusters))

    plot.figure(figsize=(20, 20))
    plot.title(f'K-means (K={k}, {"" if successful else "un"}successful)')
    plot.xlim(-5000, 5000)
    plot.ylim(-5000, 5000)
    plot.xlabel('X-axis')
    plot.ylabel('Y-axis')

    # Cluster plotting and visualization
    for i, cluster in enumerate(clusters):
        centroid = centroids[i]
        c_x, c_y = centroid

        print(f"Cluster #{i + 1}: {calculate_avg_distance_from_center(cluster, centroid)}")

        x_coords = [t[0] for t in cluster]
        y_coords = [t[1] for t in cluster]

        plot.scatter(x_coords, y_coords)
        plot.scatter(c_x, c_y, color='black' if calculate_avg_distance_from_center(cluster, centroid) <= 500 else 'red')

    plot.savefig('k-means.png', dpi=300, bbox_inches='tight')
    plot.show()


def k_medoids_visualize(points, k):
    print(f'Attempting K-medoids (K = {k})...')

    clusters, medoids, time_secs = k_medoids(points, k)

    successful = not any(calculate_avg_distance_from_center(cl, medoids[i]) > 500 for i, cl in enumerate(clusters))

    print(f"Execution time: {time_secs}s")

    plot.figure(figsize=(20, 20))
    plot.title(f'K-medoids (K={k}, {"" if successful else "un"}successful)')
    plot.xlim(-5000, 5000)
    plot.ylim(-5000, 5000)
    plot.xlabel('X-axis')
    plot.ylabel('Y-axis')

    # Cluster plotting and visualization
    for i, cluster in enumerate(clusters):
        medoid = medoids[i]
        m_x, m_y = medoid

        print(f"Cluster #{i + 1}: {calculate_avg_distance_from_center(cluster, medoid)}")

        x_coords = [t[0] for t in cluster]
        y_coords = [t[1] for t in cluster]

        plot.scatter(x_coords, y_coords)
        plot.scatter(m_x, m_y, color='black' if calculate_avg_distance_from_center(cluster, medoid) <= 500 else 'red')

    plot.savefig('k_medoids.png', dpi=300, bbox_inches='tight')
    plot.show()


def divisive_clustering_visualize(points):
    print(f'Attempting divisive clustering...')

    clusters, successful, time_secs = divisive_clustering(points)
    centroids = [calculate_centroid(cluster) for cluster in clusters]

    print(f"Execution time: {time_secs}s")

    plot.figure(figsize=(20, 20))
    plot.title(f'Divisive clustering (centroid, {"" if successful else "un"}successful)')
    plot.xlim(-5000, 5000)
    plot.ylim(-5000, 5000)
    plot.xlabel('X-axis')
    plot.ylabel('Y-axis')

    # Cluster plotting and visualization
    for i, cluster in enumerate(clusters):
        centroid = centroids[i]
        c_x, c_y = centroid

        print(f"Cluster #{i + 1}: {calculate_avg_distance_from_center(cluster, centroid)}")

        x_coords = [t[0] for t in cluster]
        y_coords = [t[1] for t in cluster]

        plot.scatter(x_coords, y_coords)
        plot.scatter(c_x, c_y, color='black' if calculate_avg_distance_from_center(cluster, centroid) <= 500 else 'red')

    plot.savefig('divisive_clustering.png', dpi=300, bbox_inches='tight')
    plot.show()


if __name__ == '__main__':
    data = gen_20_points()
    gen_other_40000_points(data)

    k_means_visualize(data, 20)
    k_medoids_visualize(data, 20)
    divisive_clustering_visualize(data)
