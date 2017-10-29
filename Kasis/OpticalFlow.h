#pragma once
#include "opencv3-2.h"

class opticalFlow {
	std::vector<Point2f> _oldPoints;
	std::vector<Point2f> _newPoints;
	std::vector<unsigned char> status;
	std::vector<float> errors;
	Mat _oldFrame;

	Size _winSize;

	int _maxLevel;

public:
	opticalFlow( Size winSize, int maxLevel,Mat frame, std::vector<Point2f> points);

	void run(Mat *frame, std::vector<Point2f> nowPoints);

	void drawCornerInImage(Mat* src, int r, Scalar rgb);

	bool isChanged(double threshold);

	std::vector<Point2f> getPoint();
};

class SelfOpticalFlow {
private:
	Matrix<int,Dynamic,Dynamic> _oldImage;
	
public:
	SelfOpticalFlow(Mat* image);
	
	std::vector<Point2f> Calc(Mat* nextImage, std::vector<Point2f> point);

};