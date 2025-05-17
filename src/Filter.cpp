#include "Filter.h"
#include <algorithm>
#include "glm/glm.hpp"
#include <queue>
#include <unordered_map>

std::vector<Region> Filter::regions;

void Filter::colorToGrayscale(cv::Mat colorImage) {
	int width = colorImage.cols;
	int height = colorImage.rows;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			cv::Vec4b color = colorImage.at<cv::Vec4b>(y, x);
			uchar gray = (uchar)((float)color[0] * 0.114 + (float)color[1] * 0.587 + (float)color[2] * 0.299);
			colorImage.at<cv::Vec4b>(y, x) = cv::Vec4b(gray, gray, gray, color[3]);
		}
	}
}

void Filter::colorToGrayscale(cv::Vec4b* colorData, int width, int height) {
	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			//offset = y * width + x;
			cv::Vec4b color = colorData[offset];
			uchar gray = (color[0] + color[1] + color[2]) / 3;
			colorData[offset] = cv::Vec4b(gray, gray, gray, color[3]);
			offset++;
		}
	}
}

void Filter::filterColorAverage(cv::Vec4b* colorData, cv::Vec4b* colorProcessedData, int width, int height) {
	int offset, offset_neighbor;
	for (int y = 2; y < height - 2; y++)
	{
		for (int x = 2; x < width - 2; x++)
		{
			cv::Vec4f color = cv::Vec4f(0, 0, 0, 0);
			for (int k = -2; k <= 2; k++)
			{
				for (int l = -2; l <= 2; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					cv::Vec4b color_neighbor = colorData[offset_neighbor];
					color += (cv::Vec4f)color_neighbor;
				}
			}

			color /= 25;
			offset = y * width + x;
			colorProcessedData[offset] = cv::Vec4b(color[0], color[1], color[2], color[3]);
		}
	}
}

void Filter::filterDepthGaussian(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {
	int offset, offset_neighbor;
	int gaussKernel[5][5] = { {1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1} };
	for (int y = 2; y < height - 2; y++)
	{
		for (int x = 2; x < width - 2; x++)
		{
			cv::Vec4f color(0, 0, 0, 0);
			for (int k = -2; k <= 2; k++)
			{
				for (int l = -2; l <= 2; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					color += (cv::Vec4f)depthData[offset_neighbor] * gaussKernel[k + 2][l + 2];
				}
			}

			color /= 273;
			offset = y * width + x;
			depthProcessedData[offset] = (cv::Vec4b)color;
		}
	}
}

void Filter::filterGrayscaleGaussian(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {
	int offset, offset_neighbor;
	int gaussKernel[3][3] = { {1,2,1},{2,4,2},{1,2,1} };
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float gray = 0;
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					gray += (float)grayscaleData[offset_neighbor] * gaussKernel[k + 1][l + 1];
				}
			}

			gray /= 16;
			offset = y * width + x;
			grayscaleProcessedData[offset] = (uchar)gray;
		}
	}
}

void Filter::filterGrayscaleSobel(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {
	int offset, offset_neighbor;
	int sobelKernelX[3][3] = { {1,0,-1},{2,0,-2},{1,0,-1} }, sobelKernelY[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float grayX = 0, grayY = 0;
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					grayX += (float)grayscaleData[offset_neighbor] * sobelKernelX[k + 1][l + 1];
					grayY += (float)grayscaleData[offset_neighbor] * sobelKernelY[k + 1][l + 1];
				}
			}

			offset = y * width + x;
			grayscaleProcessedData[offset] = (uchar)sqrt(grayX * grayX + grayY * grayY);
		}
	}
}

void Filter::filterGrayscaleMedianFilter(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {
	int offset, offset_neighbor;
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			uchar neighbors[9];
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					neighbors[3 * (k + 1) + (l + 1)] = grayscaleData[offset_neighbor];
				}
			}
			std::sort(neighbors, neighbors + 9);
			offset = y * width + x;
			grayscaleProcessedData[offset] = neighbors[4];
		}
	}
}

void Filter::filterDepthByDistance(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {
	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			cv::Vec4b color = depthData[offset];
			for (int i = 0;i < 4;i++) {
				color[i] = color[i] - color[i] % 50;
			}
			depthProcessedData[offset] = cv::Vec4b(color[0], color[1], color[2], color[3]);
			offset++;
		}
	}
}

void Filter::filterDepthPrewitt(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {
	int offset, offset_neighbor;
	int prewittKernelX[3][3] = { {1,0,-1},{1,0,-1},{1,0,-1} }, prewittKernelY[3][3] = { {1,1,1},{0,0,0},{-1,-1,-1} };
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float grayX = 0, grayY = 0;
			uchar gray;
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					grayX += (float)depthData[offset_neighbor][0] * prewittKernelX[k + 1][l + 1];
					grayY += (float)depthData[offset_neighbor][0] * prewittKernelY[k + 1][l + 1];
				}
			}

			offset = y * width + x;
			gray = (uchar)sqrt(grayX * grayX + grayY * grayY);
			depthProcessedData[offset] = cv::Vec4b(gray, gray, gray, depthData[offset][3]);
		}
	}
}


void Filter::computeNormals(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasureComputedData, int width, int height) {
	glm::vec3 p_left_vec, p_right_vec, p_up_vec, p_down_vec;
	cv::Vec4f p_left, p_right, p_up, p_down;
	glm::vec3 vec_horiz, vec_vert;
	glm::vec3 normal;

	int offset;
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			offset = y * width + x;
			p_left = pointCloudData[offset - 1];
			p_right = pointCloudData[offset + 1];
			p_up = pointCloudData[offset - width];
			p_down = pointCloudData[offset + width];
			p_left_vec = glm::vec3(p_left[0], p_left[1], p_left[2]);
			p_right_vec = glm::vec3(p_right[0], p_right[1], p_right[2]);
			p_up_vec = glm::vec3(p_up[0], p_up[1], p_up[2]);
			p_down_vec = glm::vec3(p_down[0], p_down[1], p_down[2]);
			vec_horiz = p_right_vec - p_left_vec;
			vec_vert = p_up_vec - p_down_vec;
			normal = glm::cross(vec_horiz, vec_vert);
			if (glm::length(normal) > 0.00001)
				normal = glm::normalize(normal);
			normalMeasureComputedData[offset] = cv::Vec4f(normal.x, normal.y, normal.z, 1);
		}
	}
}


void Filter::transformNormalsToImage(cv::Vec4f* normalMeasureComputedData, cv::Vec4b* normalImageComputedData, int width, int height)
{

	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			normalImageComputedData[offset] = cv::Vec4b((abs(normalMeasureComputedData[offset][2]) + 1) / 2 * 255,
				(abs(normalMeasureComputedData[offset][1]) + 1) / 2 * 255,
				(abs(normalMeasureComputedData[offset][0]) + 1) / 2 * 255, 255);

			offset++;
		}
	}
}

void Filter::computeNormals5x5Vicinity(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasureComputedData, int width, int height) {
	cv::Vec4f p_left, p_right, p_up, p_down;
	glm::vec3 vec_horiz, vec_vert;
	glm::vec3 normal;

	int offset;
	for (int y = 2; y < height - 2; y++)
	{
		for (int x = 2; x < width - 2; x++)
		{
			vec_horiz = glm::vec3(0, 0, 0);
			vec_vert = glm::vec3(0, 0, 0);
			offset = y * width + x;
			for (int k = -2;k <= 2;k++) {
				p_left = pointCloudData[offset + k * width - 2];
				p_right = pointCloudData[offset + k * width + 2];
				p_up = pointCloudData[offset + k - 2 * width];
				p_down = pointCloudData[offset + k + 2 * width];
				vec_horiz += glm::vec3(p_right[0], p_right[1], p_right[2]) - glm::vec3(p_left[0], p_left[1], p_left[2]);
				vec_vert += glm::vec3(p_up[0], p_up[1], p_up[2]) - glm::vec3(p_down[0], p_down[1], p_down[2]);
			}
			vec_horiz /= 5;
			vec_vert /= 5;
			normal = glm::cross(vec_horiz, vec_vert);
			if (glm::length(normal) > 0.00001)
				normal = glm::normalize(normal);
			normalMeasureComputedData[offset] = cv::Vec4f(normal.x, normal.y, normal.z, 1);
		}
	}
}

void Filter::filterNormalSobel(cv::Vec4f* normalMeasureData, cv::Vec4b* normalMeasureProcessedData, int width, int height) {
	cv::Vec4f normal_left, normal_right, normal_up, normal_down;

	int offset;
	int sobelKernel[3] = { 1,2,1 };
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float grayX = 0, grayY = 0;
			uchar gray;
			offset = y * width + x;
			for (int k = -1; k <= 1; k++)
			{
				normal_left = normalMeasureData[offset + k * width - 1];
				normal_right = normalMeasureData[offset + k * width + 1];
				normal_up = normalMeasureData[offset + k - width];
				normal_down = normalMeasureData[offset + k + width];
				grayX += (1 - glm::dot(glm::vec3(normal_left[0], normal_left[1], normal_left[2]), glm::vec3(normal_right[0], normal_right[1], normal_right[2]))) * sobelKernel[k + 1];
				grayY += (1 - glm::dot(glm::vec3(normal_up[0], normal_up[1], normal_up[2]), glm::vec3(normal_down[0], normal_down[1], normal_down[2]))) * sobelKernel[k + 1];
			}
			if (isnan(grayX) || isnan(grayY)) {
				gray = 0;
			}
			else {
				gray = (uchar)std::min(255.0f, 2550 * sqrt(grayX * grayX + grayY * grayY));
			}
			normalMeasureProcessedData[offset] = cv::Vec4b(gray, gray, gray, 255);
		}
	}
}

void Filter::filterDepthSobel(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {
	int offset, offset_neighbor;
	int sobelKernelX[3][3] = { {1,0,-1},{2,0,-2},{1,0,-1} }, sobelKernelY[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			float grayX = 0, grayY = 0;
			uchar gray;
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					grayX += (float)depthData[offset_neighbor][0] * sobelKernelX[k + 1][l + 1];
					grayY += (float)depthData[offset_neighbor][0] * sobelKernelY[k + 1][l + 1];
				}
			}

			offset = y * width + x;
			gray = (uchar)sqrt(grayX * grayX + grayY * grayY);
			depthProcessedData[offset] = cv::Vec4b(gray, gray, gray, depthData[offset][3]);
		}
	}
}
void Filter::filterCombinedSobel(uchar* grayscaleProcessedData, cv::Vec4b* depthProcessedData, cv::Vec4b* normalMeasureProcessedData, uchar* combinedSobelImageData, int width, int height) {
	for (int offset = 0;offset < width * height;offset++) {
		combinedSobelImageData[offset] = (uchar)std::min(255.0, 0.4 * grayscaleProcessedData[offset] + depthProcessedData[offset][0] + normalMeasureProcessedData[offset][0]);
	}
}
void Filter::filterBinarization(uchar* sobelImageData, uchar* binarizedSobelImageData, int width, int height) {
	for (int offset = 0;offset < width * height;offset++) {
		if (sobelImageData[offset] < 25) {
			binarizedSobelImageData[offset] = 0;
		}
		else {
			binarizedSobelImageData[offset] = 255;
		}
	}
}
void Filter::filterDilation(uchar* binarizedSobelImageData, uchar* dilatedBinarizedSobelImageData, int width, int height) {
	int offset = 0, offset_neighbor;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (binarizedSobelImageData[offset] == 0) {
				dilatedBinarizedSobelImageData[offset] = 0;
				for (int k = -1; k <= 1; k++)
				{
					for (int l = -1; l <= 1; l++)
					{
						if (y + k >= 0 && y + k < height && x + l >= 0 && x + l < width) {
							offset_neighbor = offset + k * width + l;
							if (binarizedSobelImageData[offset_neighbor] != 0) {
								dilatedBinarizedSobelImageData[offset] = 255;
								goto exit;
							}
						}
					}
				}
			}
			else {
				dilatedBinarizedSobelImageData[offset] = 255;
			}
		exit:
			offset++;
		}
	}
}
void Filter::edgeSegmentation(uchar* edgeImageData, ushort* regionsData, int width, int height) {
	std::queue<int> region;
	int offset, offset_neighbor, n = 1;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			offset = y * width + x;
			if (edgeImageData[offset] != 255 && regionsData[offset] == 0) {
				region.push(offset);
				regionsData[offset] = n;
				while (!region.empty()) {
					offset = region.front();
					for (int k = -1;k <= 1;k++) {
						for (int l = -1;l <= 1;l++) {
							offset_neighbor = offset + k * width + l;
							if ((offset / width + k >= 0 && offset / width + k < height && offset % width + l >= 0 && offset % width + l < width) &&
								edgeImageData[offset_neighbor] != 255 && regionsData[offset_neighbor] == 0) {
								region.push(offset_neighbor);
								regionsData[offset_neighbor] = n;
							}
						}
					}
					region.pop();
				}
				n++;
			}
		}
	}
}
void Filter::regionsToRandomColorImage(ushort* regionsData, cv::Vec4b* segmentedImageRandomColorData, int width, int height) {
	int n = *std::max_element(regionsData, regionsData + width * height) + 1;
	static std::vector<cv::Vec4b> colors;
	if (colors.size() < n) {
		int n_old = colors.size();
		colors.resize(n);
		srand(time(0));
		for (int i = n_old; i < n; i++) {
			colors[i] = cv::Vec4b(rand() % 256, rand() % 256, rand() % 256, 255);
		}
	}
	for (int offset = 0;offset < width * height;offset++) {
		segmentedImageRandomColorData[offset] = colors[regionsData[offset]];
	}
}

void Filter::planarSegmentation(uchar* grayscaleData, cv::Vec4b* depthData, cv::Vec4f* normalMeasureData, cv::Vec4f* pointCloudData, ushort* regionsData, int width, int height) {
	int offset = 0, offset_neighbor, offset_min, n = 0;
	float cost, cost_normal, cost_depth, cost_min;
	regions.clear();
	for (int y = 0;y < height;y++)
	{
		for (int x = 0;x < width;x++)
		{
			cv::Vec4f& normal = normalMeasureData[offset];
			offset_min = -1;
			cost_min = 350;
			for (int k = -1;k <= 0;k++) {
				for (int l = -1;l <= 1;l++) {
					offset_neighbor = offset + k * width + l;
					if ((y + k >= 0 && y + k < height && x + l >= 0 && x + l < width) && offset_neighbor < offset) {
						Region& region = regions[regionsData[offset_neighbor]];
						if (isnan(normal[0]) || isnan(normalMeasureData[offset_neighbor][0])) {
							cost_normal = 0;
							cost_depth = 0;
						}
						else {
							cost_normal = abs(1 - glm::dot(glm::vec3(normal[0], normal[1], normal[2]), glm::vec3(region.mean_normal[0], region.mean_normal[1], region.mean_normal[2])));
							cost_depth = abs(depthData[offset][0] - depthData[offset_neighbor][0]);
						}
						cost = abs(grayscaleData[offset] - region.mean_intensity) + 1000 * cost_normal + 50 * cost_depth;
						if (cost < cost_min) {
							offset_min = offset_neighbor;
							cost_min = cost;
						}
					}
				}
			}
			if (offset_min != -1) {
				regions[regionsData[offset_min]].add_pixel(grayscaleData[offset], depthData[offset][0], normal, pointCloudData[offset]);
				regionsData[offset] = regionsData[offset_min];
			}
			else {
				regions.push_back(Region(grayscaleData[offset], depthData[offset][0], normal, pointCloudData[offset]));
				regionsData[offset] = n;
				n++;
			}
			offset++;
		}
	}
	offset = 0;
	for (int y = 0;y < height;y++)
	{
		for (int x = 0;x < width;x++)
		{
			for (int k = -1;k <= 1;k++) {
				for (int l = -1;l <= 1;l++) {
					offset_neighbor = offset + k * width + l;
					if ((y + k >= 0 && y + k < height && x + l >= 0 && x + l < width) && regionsData[offset_neighbor] != regionsData[offset]) {
						regions[regionsData[offset]].neighbors.insert(regionsData[offset_neighbor]);
					}
				}
			}
			offset++;
		}
	}
}
void Filter::regionMerging(ushort* regionsData, int width, int height) {
	float cost, cost_distance;
	std::unordered_map<ushort, ushort> ids;
	std::unordered_set<ushort> merged_neighbors;
	glm::vec3 normal, normal_neighbor;
	for (ushort i = 0;i < regions.size();i++) {
		ids[i] = i;
	}
	for (ushort i = 0;i < ids.size();i++) {
		normal = glm::vec3(regions[ids[i]].mean_normal[0], regions[ids[i]].mean_normal[1], regions[ids[i]].mean_normal[2]);
		merged_neighbors.clear();
		for (ushort neighbor : regions[ids[i]].neighbors) {
			if (ids[neighbor] != ids[i]) {
				normal_neighbor = glm::vec3(regions[ids[neighbor]].mean_normal[0], regions[ids[neighbor]].mean_normal[1], regions[ids[neighbor]].mean_normal[2]);
				if (glm::length(normal) < 1e-2 || glm::length(normal_neighbor) < 1e-2) {
					cost_distance = 0;
				}
				else {
					cost_distance = abs(regions[ids[i]].distance() - regions[ids[neighbor]].distance());
				}
				cost = cost_distance + 4000 * abs(1 - glm::dot(normal, normal_neighbor));
				if (cost < 1400) {
					merged_neighbors.insert(neighbor);
				}
			}
		}
		for (ushort neighbor : merged_neighbors) {
			if (ids[neighbor] != ids[i]) {
				regions[ids[i]].merge_region(regions[ids[neighbor]]);
				regions.erase(regions.begin() + ids[neighbor]);
				for (ushort j = 0;j < ids.size();j++) {
					if (ids[j] == ids[neighbor] && j != neighbor) {
						ids[j] = ids[i];
					}
				}
				for (ushort j = 0;j < ids.size();j++) {
					if (ids[j] > ids[neighbor]) {
						ids[j]--;
					}
				}
				ids[neighbor] = ids[i];
			}
		}
	}
	for (int offset = 0;offset < width * height;offset++) {
		regionsData[offset] = ids[regionsData[offset]];
	}
}
void Filter::regionsToPropertyImages(ushort* regionsData, uchar* segmentedImageGrayscaleData, cv::Vec4b* segmentedImageDepthData, cv::Vec4b* segmentedImageNormalData, int width, int height) {
	for (int offset = 0;offset < width * height;offset++) {
		Region& region = regions[regionsData[offset]];
		segmentedImageGrayscaleData[offset] = region.mean_intensity;
		segmentedImageDepthData[offset] = cv::Vec4b(region.mean_depth, region.mean_depth, region.mean_depth, 255);
		segmentedImageNormalData[offset] = cv::Vec4b((abs(region.mean_normal[2]) + 1) / 2 * 255, (abs(region.mean_normal[1]) + 1) / 2 * 255, (abs(region.mean_normal[0]) + 1) / 2 * 255, 255);
	}
}

void Filter::labelRegions(ushort* regionsData, cv::Vec4b* labeledImageData, int width, int height) {
	const cv::Vec4b labelColors[4] = { cv::Vec4b(0, 255, 0, 255),cv::Vec4b(255, 0, 0, 255),cv::Vec4b(0, 255, 255, 255),cv::Vec4b(0, 0, 255, 255) };
	for (Region& region : regions) {
		glm::vec3 normal(region.mean_normal[0], region.mean_normal[1], region.mean_normal[2]);
		if (glm::dot(normal, glm::vec3(0, -1, 0)) > 0.7 && region.distance() > 2000) {
			region.label = FLOOR;
		}
		else if (abs(glm::dot(normal, glm::vec3(0, -1, 0))) < 0.5 && region.n > 5000) {
			region.label = WALL;
		}
		else if (glm::dot(normal, glm::vec3(0, 1, 0)) > 0.3) {
			region.label = CEILING;
		}
		else {
			region.label = GENERIC_OBJECT;
		}
	}
	for (int offset = 0;offset < width * height;offset++) {
		LabelType& label = regions[regionsData[offset]].label;
		if (label != UNLABELED) {
			labeledImageData[offset] = labelColors[label];
		}
	}
}
