/*********************************************
 * Author: Bo Sun                            *
 * Afflication: TAMS, University of Hamburg  *
 * E-Mail: bosun@informatik.uni-hamburg.de   *
 *         user_mail@QQ.com                  *
 * Date: Nov 13, 2014                        *
 * Licensing: GNU GPL license.               *
 *********************************************/

/** \brief TestPhaseCorrelation tests the PhaseCorrelation1D,
  * PhaseCorrelation2D and PhaseCorrelation3D library.
  *
  * Main purpose is to test the application 'receptor', in other words,
  * we should apply the result to which input to get another input?
  *
  * For OUR PhaseCorrelation1D & PhaseCorrelation2D &PhaseCorrelation3D,
  * The result is that we should apply the result to "pattern"
  * (second parameter) to match "signal"(the first parameter)
  *
  * FOR PhaseCorrelation2D/PhaseCorrelation3D, the code works
  * accord with theoretical induction.
  * For Phasecorrelation1D, we did a little trick to let it work
  * accord with PhaseCorrelation2D/PhaseCorrelation3D
  */
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "fftw3.h"
#include "PhaseCorrelation.h"
#include "PhaseCorrelation.hpp"

#ifndef TYPE_DEFINITION_
#define TYPE_DEFINITION_
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixRowXf;
#endif /*TYPE_DEFINITION_*/

int
main()
{
    // Test based on 1D vector
    // Generate a vector whose size is 'vector_size'
    int vector_size = 200;
    Eigen::VectorXf object_vector = Eigen::VectorXf::Zero(vector_size);
    for (int i=0; i<vector_size; i++)
    {
        object_vector(i) = std::rand()%255;
    }

    // Generate scene_vector by shifting object_vector to RIGHT by offset
    Eigen::VectorXf scene_vector = Eigen::VectorXf::Zero(vector_size);
    int offset =50;
    for (int j=0; j <vector_size; j++)
    {
        if (j-offset <0)
            scene_vector(j) = object_vector(j-offset+vector_size);
        else if(j-offset >=vector_size)
            scene_vector(j) = object_vector(j-offset-vector_size);
        else
            scene_vector(j) = object_vector(j-offset);
    }

    // Phase Correlation
    int vector_result;
    PhaseCorrelation1D(object_vector, scene_vector,
                       vector_size, vector_result);

    std::cout << std::endl;
    std::cout << "--------------------------------------" <<std::endl;
    std::cout << "*Phase Correlation based on 1D vector*" << std::endl;
    std::cout << std::endl;
    std::cout << "We should shift (right) object_vector by " <<offset << std::endl;
    std::cout << "to get scene_object" << std::endl;
    std::cout << "The PhaseCorrelation1D tell us to shift" << std::endl;
    std::cout << "(right) scene_object by " << vector_result <<std::endl;
    std::cout << "to get object_object" << std::endl;
    std::cout << "--------------------------------------" <<std::endl;

    // Test based on 2D image
    // Generate a 2D image whose size is image_size
    Eigen::Vector2i image_size;
    image_size << 200,300;
    EigenMatrixRowXf object_img = EigenMatrixRowXf::Zero(image_size(0), image_size(1));

    for (int i=0; i<image_size(0); i++)
    {
        for (int j=0; j<image_size(1); j++)
        {
            object_img(i,j) = std::rand()%255;
        }
    }

    // Generate scene_img by
    // shifting object_image to DOWN  by translation(0)
    // shifting object_image to RIGHT by translation(1)
    Eigen::Vector2i translation;
    translation << -25, 125;
    EigenMatrixRowXf scene_img  = EigenMatrixRowXf::Zero(image_size(0), image_size(1));
    int x_index, y_index;
    for (int i=0; i<image_size(0); i++)
    {
        for (int j=0; j<image_size(1); j++)
        {
            if (i-translation(0)<0)
                x_index = i-translation(0)+image_size(0);
            else if (i-translation(0)>=image_size(0))
                x_index = i-translation(0)-image_size(0);
            else
                x_index = i-translation(0);

            if (j-translation(1)<0)
                y_index = j-translation(1)+image_size(1);
            else if (j-translation(1)>=image_size(1))
                y_index = j-translation(1)-image_size(1);
            else
                y_index = j-translation(1);

            scene_img(i,j) = object_img(x_index, y_index);
        }
    }

    // Phase Correlation
    Eigen::Vector2i image_result;
    PhaseCorrelation2D(object_img, scene_img,
                       image_size(0), image_size(1),
                       image_result(0),image_result(1));

    std::cout << std::endl;
    std::cout << "--------------------------------------" <<std::endl;
    std::cout << "*Phase Correlation based on 2D image*" << std::endl;
    std::cout << std::endl;
    std::cout << "We should shift (right) object_image by " <<translation(1) << std::endl;
    std::cout << "and then shift (down) object_image by " << translation(0) << std::endl;
    std::cout << "to get scene_image" << std::endl;
    std::cout << "The PhaseCorrelation2D tell us to shift" << std::endl;
    std::cout << "(right) scene_image by " << image_result(1) <<std::endl;
    std::cout << "and then shift(down) scene_image by " << image_result(0) << std::endl;
    std::cout << "to get object_image" << std::endl;
    std::cout << "--------------------------------------" <<std::endl;

    // Test based on the 1D vector
    // generated by summing the row of 2D image
    Eigen::VectorXf object_img_sum_row;
    Eigen::VectorXf scene_img_sum_row;
    object_img_sum_row = object_img.rowwise().sum();
    scene_img_sum_row = scene_img.rowwise().sum();

    if (object_img_sum_row.size()!=image_size(0)||
            scene_img_sum_row.size()!=image_size(0))
        std::cout << "Sum of rows wrong!" << std::endl;

    // Phase Correlation
    int result_sum_row;
    PhaseCorrelation1D(object_img_sum_row, scene_img_sum_row,
                       image_size(0), result_sum_row);

    std::cout << std::endl;
    std::cout << "--------------------------------------" <<std::endl;
    std::cout << "*Phase Correlation based on 1D vector generated "<<std::endl;
    std::cout << "by summing the row of 2D image*" << std::endl;
    std::cout << std::endl;
    std::cout << "We should shift (down) object_img_sum_row by " <<translation(0) << std::endl;
    std::cout << "to get scene_img_sum_row" << std::endl;
    std::cout << "The PhaseCorrelation1D tell us to shift" << std::endl;
    std::cout << "(down) scene_img_sum_row by " << result_sum_row <<std::endl;
    std::cout << "to get object_img_sum_row" << std::endl;
    std::cout << "--------------------------------------" <<std::endl;

    // Test based on the 1D vector
    // generated by summing the column of 2D image
    Eigen::VectorXf object_img_sum_col;
    Eigen::VectorXf scene_img_sum_col;
    object_img_sum_col = object_img.colwise().sum();
    scene_img_sum_col  = scene_img.colwise().sum();
    if (object_img_sum_col.size()!=image_size(1)||
            scene_img_sum_col.size()!=image_size(1))
        std::cout << "Sum of columns wrong!" << std::endl;

    // Phase Correlation
    int result_sum_col;
    PhaseCorrelation1D(object_img_sum_col, scene_img_sum_col,
                       image_size(1),result_sum_col);
    std::cout << std::endl;
    std::cout << "--------------------------------------" <<std::endl;
    std::cout << "*Phase Correlation based on 1D vector generated "<<std::endl;
    std::cout << "by summing the column of 2D image*" << std::endl;
    std::cout << std::endl;
    std::cout << "We should shift (right) object_img_sum_col by " <<translation(1) << std::endl;
    std::cout << "to get scene_img_sum_col" << std::endl;
    std::cout << "The PhaseCorrelation1D tell us to shift" << std::endl;
    std::cout << "(wright) scene_img_sum_col by " << result_sum_col <<std::endl;
    std::cout << "to get object_img_sum_col" << std::endl;
    std::cout << "--------------------------------------" <<std::endl;

    // Test based on 3D volume
    Eigen::Vector3i volume_size;
    volume_size << 100, 120, 150;
    float *object_volume;
    float *scene_volume;
    object_volume = (float*) calloc (volume_size(0)*volume_size(1)*volume_size(2),
                                     sizeof(float));
    scene_volume  = (float*) calloc (volume_size(0)*volume_size(1)*volume_size(2),
                                     sizeof(float));
    if (object_volume==NULL||scene_volume==NULL)
        return(1);
    // Generate 3D volume whose size is volume_size
    for (int i=0; i<volume_size(0)*volume_size(1)*volume_size(2); i++)
        object_volume[i] = std::rand()%255;

    // Generate scene volume by
    // shifting object volume DOWN by translation_volume(0)
    // shifting object volume RIGHT by translation_volume(1)
    // shifting object volume CLOSE TO VIEWER by translation_volume(2)
    // storage order ---> row major
    Eigen::Vector3i translation_volume;
    translation_volume << 10, 12, 15;
    int volume_x_index, volume_y_index, volume_z_index;
    for (int i=0; i<volume_size(0);i++)
    {
        for (int j=0; j<volume_size(1); j++)
        {
            for (int k=0; k<volume_size(2); k++)
            {
                if (i-translation_volume(0)<0)
                    volume_x_index = i-translation_volume(0)+volume_size(0);
                else if (i-translation_volume(0)>=volume_size(0))
                    volume_x_index = i-translation_volume(0)-volume_size(0);
                else
                    volume_x_index = i-translation_volume(0);

                if (j-translation_volume(1)<0)
                    volume_y_index = j-translation_volume(1)+volume_size(1);
                else if (j-translation_volume(1)>=volume_size(1))
                    volume_y_index = j-translation_volume(1)-volume_size(1);
                else
                    volume_y_index = j-translation_volume(1);

                if (k-translation_volume(2)<0)
                    volume_z_index = k-translation_volume(2)+volume_size(2);
                else if (k-translation_volume(2)>=volume_size(2))
                    volume_z_index = k-translation_volume(2)-volume_size(2);
                else
                    volume_z_index = k-translation_volume(2);

                scene_volume[k+volume_size(2)*(j+volume_size(1)*i)] =
                        object_volume[volume_z_index+volume_size(2)*(volume_y_index+volume_size(1)*volume_x_index)];
            }
        }
    }
    Eigen::Vector3i volume_result;
    PhaseCorrelation3D(object_volume, scene_volume,
                       volume_size(0), volume_size(1), volume_size(2),
                       volume_result(0), volume_result(1), volume_result(2));

    std::cout << std::endl;
    std::cout << "--------------------------------------" <<std::endl;
    std::cout << "*Phase Correlation based on 3D volume*" << std::endl;
    std::cout << std::endl;
    std::cout << "We should shift (down) object_volume by " <<translation_volume(0) << std::endl;
    std::cout << "and then shift (right) object_volume by " << translation_volume(1) << std::endl;
    std::cout << "and then shift (close to viewer) object_volume by " << translation_volume(2) << std::endl;
    std::cout << "to get scene_volume" << std::endl;
    std::cout << "The PhaseCorrelation3D tell us to shift" << std::endl;
    std::cout << "(down) scene_volume by " << volume_result(0) <<std::endl;
    std::cout << "and then shift(right) scene_volume by " << volume_result(1) << std::endl;
    std::cout << "and then shift(close to viewer) scene_volume by " << volume_result(2) << std::endl;
    std::cout << "to get object_volume" << std::endl;
    std::cout << "--------------------------------------" <<std::endl;

    return (0);

}
