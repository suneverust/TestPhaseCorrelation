/*********************************************
 * Author: Bo Sun                            *
 * Afflication: TAMS, University of Hamburg  *
 * E-Mail: bosun@informatik.uni-hamburg.de   *
 *         user_mail@QQ.com                  *
 * Date: Nov 13, 2014                        *
 *********************************************/

#ifndef  TEST_PHASECORRELATION_IMPL_H_
#define  TEST_PHASECORRELATION_IMPL_H_
#include "PhaseCorrelation.h"
#include <iostream>

/** \brief PhaseCorrelation1D compute the offset between two input vectors
  * based on POMF (Phase Only Matched Filter)
  * -->> Q(k) = conjugate(S(k))/|S(k)| * R(k)/|R(k)|
  * -->> q(x) = ifft(Q(k))
  * -->> (xs,ys) = argmax(q(x))
  * Note that the storage order of FFTW is row-order, while the storage
  * order of Eigen is DEFAULT column-order.
  */
void PhaseCorrelation1D(const Eigen::VectorXf signal,
                        const Eigen::VectorXf pattern,
                        const int size,
                        int &offset)
{
    // load data
    if(signal.size() != size ||
            pattern.size() !=size)
    {
        std::cout << "The size of vector input for PhaseCorrelation wrong!" << std::endl;
        return ;
    }

    fftw_complex *signal_vector = (fftw_complex*) fftw_malloc (sizeof(fftw_complex)*size);
    fftw_complex *pattern_vector = (fftw_complex*) fftw_malloc (sizeof(fftw_complex)*size);

    for (int i=0; i < signal.size(); i++)
    {
        signal_vector[i][0] = *(signal.data()+i);
        signal_vector[i][1] = 0;
    }
    for (int j=0; j < pattern.size(); j++)
    {
        pattern_vector[j][0] = *(pattern.data()+j);
        pattern_vector[j][1] = 0;
    }

    // forward fft
    fftw_plan signal_forward_plan = fftw_plan_dft_1d(size, signal_vector, signal_vector,
                                                     FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pattern_forward_plan  = fftw_plan_dft_1d(size, pattern_vector, pattern_vector,
                                                     FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute (signal_forward_plan);
    fftw_execute (pattern_forward_plan);

    // cross power spectrum
    fftw_complex *cross_vector = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size);
    double temp;
    for (int i=0; i<size; i++)
    {
        cross_vector[i][0] = (pattern_vector[i][0]*signal_vector[i][0])-
                (pattern_vector[i][1]*(-signal_vector[i][1]));
        cross_vector[i][1] = (pattern_vector[i][0]*(-signal_vector[i][1]))+
                (pattern_vector[i][1]*signal_vector[i][0]);
        temp = sqrt(cross_vector[i][0]*cross_vector[i][0]+cross_vector[i][1]*cross_vector[i][1]);
        cross_vector[i][0] /= temp;
        cross_vector[i][1] /= temp;
    }

    // backward fft
    // FFTW computes an unnormalized transform,
    // in that there is no coefficient in front of
    // the summation in the DFT.
    // In other words, applying the forward and then
    // the backward transform will multiply the input by n.

    // BUT, we only care about the maximum of the inverse DFT,
    // so we don't need to normalize the inverse result.

    // the storage order in FFTW is row-order
    fftw_plan cross_backward_plan = fftw_plan_dft_1d(size, cross_vector, cross_vector,
                                                     FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(cross_backward_plan);

    // free memory
    fftw_destroy_plan(signal_forward_plan);
    fftw_destroy_plan(pattern_forward_plan);
    fftw_free(signal_vector);
    fftw_free(pattern_vector);

    Eigen::VectorXf cross_real = Eigen::VectorXf::Zero(size);
    for (int i=0; i < size; i++)
    {
        cross_real(i) = cross_vector[i][0];
    }
    std::ptrdiff_t max_loc;
    float unuse = cross_real.maxCoeff(&max_loc);
    offset = (int) max_loc;

    if (offset > 0.5*size)
        offset = offset-size;

}

/** \brief PhaseCorrelation2D compute the offset between two input images
  * based on POMF (Phase Only Matched Filter)
  * -->> Q(k) = conjugate(S(k))/|S(k)| * R(k)/|R(k)|
  * -->> q(x) = ifft(Q(k))
  * -->> (xs,ys) = argmax(q(x))
  * Note that the storage order of FFTW is row-order, while the storage
  * order of Eigen is default column-order.
  */
void PhaseCorrelation2D(const EigenMatrixRowXf signal,
                        const EigenMatrixRowXf pattern,
                        const int height,
                        const int width,
                        int &height_offset,
                        int &width_offset)
{
    // load data
    if (signal.size() != width*height ||
            pattern.size() !=width*height)
    {
        std::cout << "The size of image input for PhaseCorrelation wrong!" << std::endl;
        return ;
    }

    fftw_complex *signal_img = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*height*width);
    fftw_complex *pattern_img = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*height*width);

    for (int i=0; i < signal.size(); i++)
    {
        signal_img[i][0] = *(signal.data()+i);
        signal_img[i][1] = 0;
    }
    for (int j=0; j < pattern.size(); j++)
    {
        pattern_img[j][0] = *(pattern.data()+j);
        pattern_img[j][1] = 0;
    }

    // forward fft
    fftw_plan signal_forward_plan = fftw_plan_dft_2d (height, width, signal_img, signal_img,
                                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pattern_forward_plan  = fftw_plan_dft_2d (height, width, pattern_img, pattern_img,
                                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute (signal_forward_plan);
    fftw_execute (pattern_forward_plan);

    // cross power spectrum
    fftw_complex *cross_img = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*height*width);
    double temp;
    for (int i=0; i < height*width; i++)
    {
        cross_img[i][0] = (signal_img[i][0]*pattern_img[i][0])-
                (signal_img[i][1]*(-pattern_img[i][1]));
        cross_img[i][1] = (signal_img[i][0]*(-pattern_img[i][1]))+
                (signal_img[i][1]*pattern_img[i][0]);
        temp = sqrt(cross_img[i][0]*cross_img[i][0]+cross_img[i][1]*cross_img[i][1]);
        cross_img[i][0] /= temp;
        cross_img[i][1] /= temp;
    }

    // backward fft
    // FFTW computes an unnormalized transform,
    // in that there is no coefficient in front of
    // the summation in the DFT.
    // In other words, applying the forward and then
    // the backward transform will multiply the input by n.

    // BUT, we only care about the maximum of the inverse DFT,
    // so we don't need to normalize the inverse result.

    // the storage order in FFTW is row-order
    fftw_plan cross_backward_plan = fftw_plan_dft_2d(height, width, cross_img, cross_img,
                                                     FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(cross_backward_plan);

    // free memory
    fftw_destroy_plan(signal_forward_plan);
    fftw_destroy_plan(pattern_forward_plan);
    fftw_destroy_plan(cross_backward_plan);
    fftw_free(signal_img);
    fftw_free(pattern_img);

    Eigen::VectorXf cross_real = Eigen::VectorXf::Zero(height*width);
    for (int i= 0; i < height*width; i++)
    {
        cross_real(i) = cross_img[i][0];
    }

    std::ptrdiff_t max_loc;
    float unuse = cross_real.maxCoeff(&max_loc);

    height_offset =floor(((int) max_loc)/ width);
    width_offset = (int)max_loc - width*height_offset;

    if (height_offset > 0.5*height)
        height_offset = height_offset-height;
    if (width_offset  > 0.5*width)
        width_offset = width_offset-width;
}

void PhaseCorrelation3D(const float *signal,
                        const float *pattern,
                        const int height,
                        const int width,
                        const int depth,
                        int &height_offset,
                        int &width_offset,
                        int &depth_offset)
{
    int size = height*width*depth;
    fftw_complex *signal_volume = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size);
    fftw_complex *pattern_volume = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size);

    for (int i=0; i < size; i++)
    {
        signal_volume[i][0] = signal[i];
        signal_volume[i][1] = 0;
    }
    for (int j=0; j < size; j++)
    {
        pattern_volume[j][0] = pattern[j];
        pattern_volume[j][1] = 0;
    }

    // forward fft
    fftw_plan signal_forward_plan = fftw_plan_dft_3d (height, width, depth, signal_volume, signal_volume,
                                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pattern_forward_plan  = fftw_plan_dft_3d (height, width, depth, pattern_volume, pattern_volume,
                                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute (signal_forward_plan);
    fftw_execute (pattern_forward_plan);

    // cross power spectrum
    fftw_complex *cross_volume = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size);
    double temp;
    for (int i=0; i < size; i++)
    {
        cross_volume[i][0] = (signal_volume[i][0]*pattern_volume[i][0])-
                (signal_volume[i][1]*(-pattern_volume[i][1]));
        cross_volume[i][1] = (signal_volume[i][0]*(-pattern_volume[i][1]))+
                (signal_volume[i][1]*pattern_volume[i][0]);
        temp = sqrt(cross_volume[i][0]*cross_volume[i][0]+cross_volume[i][1]*cross_volume[i][1]);
        cross_volume[i][0] /= temp;
        cross_volume[i][1] /= temp;
    }

    // backward fft
    // FFTW computes an unnormalized transform,
    // in that there is no coefficient in front of
    // the summation in the DFT.
    // In other words, applying the forward and then
    // the backward transform will multiply the input by n.

    // BUT, we only care about the maximum of the inverse DFT,
    // so we don't need to normalize the inverse result.

    // the storage order in FFTW is row-order
    fftw_plan cross_backward_plan = fftw_plan_dft_3d(height, width, depth, cross_volume, cross_volume,
                                                     FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(cross_backward_plan);

    // free memory
    fftw_destroy_plan(signal_forward_plan);
    fftw_destroy_plan(pattern_forward_plan);
    fftw_destroy_plan(cross_backward_plan);
    fftw_free(signal_volume);
    fftw_free(pattern_volume);

    Eigen::VectorXf cross_real(size);

    for (int i= 0; i < size; i++)
    {
        cross_real(i) = cross_volume[i][0];
    }

    std::ptrdiff_t max_loc;
    float unuse = cross_real.maxCoeff(&max_loc);

    height_offset =floor(((int) max_loc)/ (width*depth));
    width_offset = floor(((int)max_loc - width*depth*height_offset)/depth);
    depth_offset = floor((int)max_loc-width*depth*height_offset-width_offset*depth);

    if (height_offset > 0.5*height)
        height_offset = height_offset-height;
    if (width_offset  > 0.5*width)
        width_offset = width_offset-width;
    if (depth_offset > 0.5*depth)
        depth_offset = depth_offset-depth;
}

#endif /*TEST_PHASECORRELATION_IMPL_H_*/
