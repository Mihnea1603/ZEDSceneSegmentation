#include "Region.h"
#include "glm/glm.hpp"

Region::Region(uchar intensity, uchar depth, const cv::Vec4f& normal, const cv::Vec4f& position) :mean_intensity(intensity), mean_depth(depth), mean_normal(normal), mean_position(position) {
	if (isnan(normal[0])) {
		mean_depth = 0;
		mean_normal = cv::Vec4f(0, 0, 0, 1);
		mean_position = cv::Vec4f(0, 0, 0, 1);
		n_valid = 0;
	}
}
void Region::add_pixel(uchar intensity, uchar depth, const cv::Vec4f& normal, const cv::Vec4f& position) {
	mean_intensity = (n * mean_intensity + intensity) / (n + 1);
	if (!isnan(normal[0])) {
		mean_depth = (n_valid * mean_depth + depth) / (n_valid + 1);
		mean_normal = (n_valid * mean_normal + normal) / (n_valid + 1);
		mean_position = (n_valid * mean_position + position) / (n_valid + 1);
		n_valid++;
	}
	n++;
}
void Region::merge_region(const Region& region) {
	mean_intensity = (n * mean_intensity + region.n * region.mean_intensity) / (n + region.n);
	mean_depth = (n_valid * mean_depth + region.n_valid * region.mean_depth) / (n_valid + region.n_valid);
	mean_normal = (n_valid * mean_normal + region.n_valid * region.mean_normal) / (n_valid + region.n_valid);
	mean_position = (n_valid * mean_position + region.n_valid * region.mean_position) / (n_valid + region.n_valid);
	neighbors.insert(region.neighbors.begin(), region.neighbors.end());
	n += region.n;
	n_valid += region.n_valid;
}

float Region::distance() {
	glm::vec3 normal(mean_normal[0], mean_normal[1], mean_normal[2]),
		position(mean_position[0], mean_position[1], mean_position[2]);
	return abs(glm::dot(normal, position)) / glm::length(normal);
}
