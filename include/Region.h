#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_set>

enum LabelType {
	FLOOR,
	WALL,
	CEILING,
	GENERIC_OBJECT,
	UNLABELED
};

struct Region {
	int n = 1, n_valid = 1;
	uchar mean_intensity, mean_depth;
	cv::Vec4f mean_normal, mean_position;
	std::unordered_set<ushort> neighbors;
	LabelType label = UNLABELED;
	Region(uchar intensity, uchar depth, const cv::Vec4f& normal, const cv::Vec4f& position);
	void add_pixel(uchar intensity, uchar depth, const cv::Vec4f& normal, const cv::Vec4f& position);
	void merge_region(const Region& region);
	float distance();
};