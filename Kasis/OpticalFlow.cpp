#include "OpticalFlow.h"

template <typename t_matrix>
t_matrix PseudoInverse(const t_matrix& m, const double &tolerance = 1.e-6)
{
	using namespace Eigen;
	typedef JacobiSVD<t_matrix> TSVD;
	unsigned int svd_opt(ComputeThinU | ComputeThinV);
	if (m.RowsAtCompileTime != Dynamic || m.ColsAtCompileTime != Dynamic)
		svd_opt = ComputeFullU | ComputeFullV;
	TSVD svd(m, svd_opt);
	const typename TSVD::SingularValuesType &sigma(svd.singularValues());
	typename TSVD::SingularValuesType sigma_inv(sigma.size());
	for (long i = 0; i<sigma.size(); ++i)
	{
		if (sigma(i) > tolerance)
			sigma_inv(i) = 1.0 / sigma(i);
		else
			sigma_inv(i) = 0.0;
	}
	return svd.matrixV()*sigma_inv.asDiagonal()*svd.matrixU().transpose();
}

opticalFlow::opticalFlow(Size winSize, int maxLevel, Mat frame, std::vector<Point2f> oldPoints) {
	_winSize = winSize;
	_maxLevel = maxLevel;
	_oldFrame = frame;
	_oldPoints = oldPoints;
}

void opticalFlow::run(Mat *newFrame, std::vector<Point2f> nowPoints) {
	calcOpticalFlowPyrLK(
		_oldFrame,
		*newFrame,
		_oldPoints,
		_newPoints,
		status,
		errors,
		_winSize,
		_maxLevel,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03),
		OPTFLOW_LK_GET_MIN_EIGENVALS,
		0.1
	);
	_oldPoints.clear();
	_oldPoints = nowPoints;

	_oldFrame = newFrame->clone();
}


void opticalFlow::drawCornerInImage(Mat* src, int r, Scalar rgb) {
	for (int i = 0; i < status.size(); i++) {
		if (status[i] == 1) {
			circle(*src, _newPoints[i], r, rgb, -1, 8, 0);
		}
		else {
			//circle(*src, _newPoints[i], r, Scalar(255,0,0), -1, 8, 0);
		}
	}
}

bool opticalFlow::isChanged(double threshold) {
	int changed = 0;
	for (int i = 0; i < status.size(); i++) {
		if (status[i] == 1) {
			changed++;
		}
	}
	std::cout << changed / (double)status.size() * 100 << std::endl;
	if (changed / (double)status.size() * 100 >= threshold) {
		return true;
	}
	return false;
}

std::vector<Point2f> opticalFlow::getPoint() 
{
	std::vector<Point2f> detectPoint;
	for (int i = 0; i < status.size(); i++) {
		if (status[i] == 1) {
			detectPoint.push_back(_newPoints[i]);
		}
	}

	return detectPoint;
}


/// <summary>
/// 
/// </summary>
/// <param name="image"></param>
SelfOpticalFlow::SelfOpticalFlow(Mat* image) {
	cv2eigen(*image, _oldImage);
}

/// <summary>
/// 
/// </summary>
/// <param name="image"></param>
/// <param name="pointList"></param>
/// <returns></returns>
std::vector<Point2f> SelfOpticalFlow::Calc(Mat* image, std::vector<Point2f> pointList) {

	MatrixXi nextImage;
	cv2eigen(*image, nextImage);

	std::vector<Point2f> detectPoint;

	for each (Point2f point in pointList)
	{
		int x = point.x;
		int y = point.y;

		Matrix<float, 9, 1> difference;
		Matrix<float, 9, 2> gradient;
		// 基準の座標と周囲8近傍の座標値
		Point p[9] = {
			{ x - 1, y - 1 },
			{ x    , y - 1 },
			{ x + 1, y - 1 },
			{ x - 1, y },
			{ x    , y },
			{ x + 1, y },
			{ x - 1, y + 1 },
			{ x    , y + 1 },
			{ x + 1, y + 1 }
		};

		if (x <= 1
			|| x >= nextImage.cols() - 2
			|| y <= 1
			|| y >= nextImage.rows() - 2) {
			continue;
		}

		for (int i = 0; i < 9; i++) {
			// 中心座標と8近傍の差分
			difference(i, 0) = -(nextImage(p[i].y, p[i].x) - _oldImage(p[i].y, p[i].x));

			// 勾配
			// fx = f(x,y) * -1 + f(x + 1, y)     * 1
			// fy = f(x,y) * -1 + f(x    , y + 1) * 1
			//gradient(i, 0) = nextImage(p[i].y, p[i].x) * -1 + nextImage(p[i].y, p[i].x + 1) * 1;
			//gradient(i, 1) = nextImage(p[i].y, p[i].x) * -1 + nextImage(p[i].y + 1, p[i].x) * 1;
			
			// 
			gradient(i, 0) = (nextImage(p[i].y    , p[i].x - 1) + nextImage(p[i].y, p[i].x) + nextImage(p[i].y    , p[i].x + 1)) / 3;
			gradient(i, 1) = (nextImage(p[i].y - 1, p[i].x)     + nextImage(p[i].y, p[i].x) + nextImage(p[i].y + 1, p[i].x)) / 3;
		}

		MatrixXf temp = gradient.transpose() * gradient;

		MatrixXf hoge = PseudoInverse(temp) * gradient.transpose();

		MatrixXf result = hoge * difference;
		if (result(0, 0) > 0
			|| result(1, 0) > 0) {
			detectPoint.push_back(Point2f(point.x, point.y));
		}
	}
	return detectPoint;
}
