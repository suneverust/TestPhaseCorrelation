/*********************************************
 * Author: Bo Sun                            *
 * Afflication: TAMS, University of Hamburg  *
 * E-Mail: bosun@informatik.uni-hamburg.de   *
 *         user_mail@QQ.com                  *
 * Date: Nov 13, 2014                        *
 *********************************************/
#ifndef PHASECORRELATION_H_
#define PHASECORRELATION_H_
#ifndef TYPE_DEFINITION_
#define TYPE_DEFINITION_
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixRowXf;
#endif /*TYPE_DEFINITION_*/

/** \brief PhaseCorrelation1D compute the offset between two input vectors
  * based on POMF (Phase Only Matched Filter)
  * -->> Q(k) = conjugate(S(k))/|S(k)| * R(k)/|R(k)|
  * -->> q(x) = ifft(Q(k))
  * -->> (xs,ys) = argmax(q(x))
  * Note that the storage order of FFTW is row-order, while the storage
  * order of Eigen is DEFAULT column-order.
  * Parameters:
  * [in] signal         the input(signal) vector
  * [in] pattern        the input(pattern) vector
  * [in] size           the size of input vectors
  * [out] offset        the result offset, which should be applied to pattern to match signal
  *
  */
void PhaseCorrelation1D(const Eigen::VectorXf signal,
                        const Eigen::VectorXf pattern,
                        const int size,
                        int &offset);

/** \brief PhaseCorrelation2D compute the offset between two input images
  * based on POMF (Phase Only Matched Filter)
  * -->> Q(k) = conjugate(S(k))/|S(k)| * R(k)/|R(k)|
  * -->> q(x) = ifft(Q(k))
  * -->> (xs,ys) = argmax(q(x))
  * Note that the storage order of FFTW is row-order, while the storage
  * order of Eigen is DEFAULT column-order.
  * We adopt the RIGHT-hand Cartesian coordinate system.
  * Parameters:
  * [in] signal              the input(signal) image
  * [in] pattern             the input(pattern) image
  * [in] height              the height of input images(how many rows/size of column/range of x)
  * [in] width               the width of input images (how many columns/size of row/range of y)
  * [out] height_offset      the result offset, we move down (positive x axis) pattern height_offset to match signal
  * [out] width_offset       the result offset, we move right (positive y axis) pattern width_offset to match signal
  */
void PhaseCorrelation2D(const EigenMatrixRowXf signal,
                        const EigenMatrixRowXf pattern,
                        const int height,
                        const int width,
                        int &height_offset,
                        int &width_offset);

/** \brief PhaseCorrelation3D compute the offset between two input volumes
  * based on POMF (Phase Only Matched Filter)
  * -->> Q(k) = conjugate(S(k))/|S(k)| * R(k)/|R(k)|
  * -->> q(x) = ifft(Q(k))
  * -->> (xs,ys) = argmax(q(x))
  * Note that the storage order of FFTW is row-order
  * We adopt the RIGHT-hand Cartesian coordinate system.
  * Parameters:
  * [in] signal              the input(signal) volume
  * [in] pattern             the input(pattern) volume
  * [in] height              the height of input volumes(range of x)
  * [in] width               the width of input volumes (range of y)
  * [in] depth               the depth of input volumes (range of z)
  * [out] height_offset      the result offset, we move down (positive x axis) pattern height_offset to match signal
  * [out] width_offset       the result offset, we move right (positive y axis) pattern width_offset to match signal
  * [out] depth_offset       the result offset, we move close to viewer (positive z axis) pattern depth_offset to match signal
  */
void PhaseCorrelation3D(const float *signal,
                        const float *pattern,
                        const int height,
                        const int width,
                        const int depth,
                        int &height_offset,
                        int &width_offset,
                        int &depth_offset);

#endif /*TEST_PHASECORRELATION_H_*/
