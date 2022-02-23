/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkImageRegistrationMethodv4.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkBSplineTransform.h"
#include "itkLBFGSBOptimizerv4.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSquaredDifferenceImageFilter.h"
#include "itkIdentityTransform.h"
#include "itkBSplineTransformInitializer.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "itkCommand.h"

// This class handles updates during optimization of the registration, to provide feedback
// to the user on the iteration number and value of the metric
class CommandIterationUpdate : public itk::Command
{
  public:
  
    using Self = CommandIterationUpdate;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);

  protected:
  
    CommandIterationUpdate() = default;

  public:
  
    using OptimizerType = itk::LBFGSBOptimizerv4;
    using OptimizerPointer = const OptimizerType *;

    void Execute(itk::Object * caller, const itk::EventObject & event) override
    {
      Execute((const itk::Object *)caller, event);
    }

    void Execute(const itk::Object * object, const itk::EventObject & event) override
    {
      auto optimizer = static_cast<OptimizerPointer>(object);
      
      if (!(itk::IterationEvent().CheckEvent(&event)))
      {
        return;
      }
      
      // Print iteration number and current value of metric
      std::cout << "  Iteration " << optimizer->GetCurrentIteration()+1 << ": " << optimizer->GetValue() << "\n";
    }
};

int main(int argc, char * argv[])
{
  
  // If there are too few arguments print a usage message
  if (argc < 4)
  {
    std::cout << "USAGE: BSplineRegistration fixed_image moving_image output_image\n";
    return EXIT_FAILURE;
  }

  // The image type for this application is 3 dimensional with floating point values
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using FixedImageType = itk::Image<PixelType, Dimension>;
  using MovingImageType = itk::Image<PixelType, Dimension>;
  
  // Read command line arguments
  char* fixedImPath = argv[1];      // Path to the fixed image
  char* movingImPath = argv[2];     // Path to the moving image
  char* outImPath = argv[3];        // Desired output path and file name

  // Read the fixed and moving images
  using FixedImageReaderType = itk::ImageFileReader<FixedImageType>;
  using MovingImageReaderType = itk::ImageFileReader<MovingImageType>;
  FixedImageReaderType::Pointer fixedImageReader = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  fixedImageReader->SetFileName(fixedImPath);
  movingImageReader->SetFileName(movingImPath);
  
  fixedImageReader->Update();
  movingImageReader->Update();
  
  // Store the fixed and moving images
  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
  MovingImageType::Pointer movingImage = movingImageReader->GetOutput();
  
  // Set up the BSpline transformation
  const unsigned int SpaceDimension = Dimension;
  constexpr unsigned int SplineOrder = 3;
  using CoordinateRepType = double;

  // The transform that will map the fixed image into the moving image.
  // In this case it is a 3D BSpline transformation
  using TransformType = itk::BSplineTransform<CoordinateRepType, SpaceDimension, SplineOrder>;
  
  // The BSpline transformation has many parameters so we need a suitable solver
  using OptimizerType = itk::LBFGSBOptimizerv4;

  // The metric will compare how well the two images match each other. Metric
  // types are usually parameterized by the image types as it can be seen in
  // the following type declaration. Mean squares is a direct comparison
  // between intensity values.
  using MetricType = itk::MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType>;

  // The registration method type is instantiated using the types of the
  // fixed and moving images. This class is responsible for interconnecting
  // all the components that we have described so far.
  using RegistrationType = itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType>;

  // Create components
  TransformType::Pointer outputBSplineTransform = TransformType::New();
  OptimizerType::Pointer optimizer = OptimizerType::New();
  MetricType::Pointer metric = MetricType::New();
  RegistrationType::Pointer registration = RegistrationType::New();

  // Each component is now connected to the instance of the registration method
  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);

  registration->SetFixedImage(fixedImage);
  registration->SetMovingImage(movingImage);

  // We will initialize the BSpline transformation
  using InitializerType = itk::BSplineTransformInitializer<TransformType, FixedImageType>;
  InitializerType::Pointer transformInitializer = InitializerType::New();

  // A BSpline transformation is parameterized by a 3D grid of nodes
  unsigned int numberOfGridNodesInOneDimension = 8;
  TransformType::MeshSizeType meshSize;
  meshSize.Fill(numberOfGridNodesInOneDimension - SplineOrder);

  transformInitializer->SetTransform(outputBSplineTransform);
  transformInitializer->SetImage(fixedImage);
  transformInitializer->SetTransformDomainMeshSize(meshSize);
  transformInitializer->InitializeTransform();

  // Set the initial BSpline transform to identity
  using ParametersType = TransformType::ParametersType;
  const unsigned int numberOfParameters = outputBSplineTransform->GetNumberOfParameters();
  ParametersType parameters(numberOfParameters);
  parameters.Fill(0.0);
  outputBSplineTransform->SetParameters(parameters);

  registration->SetInitialTransform(outputBSplineTransform);
  registration->InPlaceOn();

  // This will be a hierarchical registration with 3 levels of downsampling/blurring
  constexpr unsigned int numberOfLevels = 3;

  // Downsample 4, 2, and finally optimize on the full resolution image
  RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize(numberOfLevels);
  shrinkFactorsPerLevel[0] = 4;
  shrinkFactorsPerLevel[1] = 2;
  shrinkFactorsPerLevel[2] = 1;

  // Blur with a Gaussian with std dev. 2, 1, and no blurring at the full resolution
  RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize(numberOfLevels);
  smoothingSigmasPerLevel[0] = 2;
  smoothingSigmasPerLevel[1] = 1;
  smoothingSigmasPerLevel[2] = 0;

  registration->SetNumberOfLevels(numberOfLevels);
  registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
  registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

  // Set up parameters for the optimizer
  const unsigned int numParameters = outputBSplineTransform->GetNumberOfParameters();
  OptimizerType::BoundSelectionType boundSelect(numParameters);
  OptimizerType::BoundValueType upperBound(numParameters);
  OptimizerType::BoundValueType lowerBound(numParameters);

  boundSelect.Fill(OptimizerType::UNBOUNDED);
  upperBound.Fill(0.0);
  lowerBound.Fill(0.0);

  optimizer->SetBoundSelection(boundSelect);
  optimizer->SetUpperBound(upperBound);
  optimizer->SetLowerBound(lowerBound);

  optimizer->SetCostFunctionConvergenceFactor(1e+12);
  optimizer->SetGradientConvergenceTolerance(1.0e-35);
  optimizer->SetNumberOfIterations(500);
  optimizer->SetMaximumNumberOfFunctionEvaluations(500);
  optimizer->SetMaximumNumberOfCorrections(5);
  
  // For updating the user on the status of the optimization
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver(itk::IterationEvent(), observer);

  std::cout << "\nBeginning registration...\n\n";

  // Start the registration optimization
  try
  {
    registration->Update();
  }
  catch (const itk::ExceptionObject & err)
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  // Get the final value of the BSpline parameters
  OptimizerType::ParametersType finalParameters = outputBSplineTransform->GetParameters();

  std::cout << "\nResult = " << std::endl;
  std::cout << "  Final parameters = " << finalParameters << std::endl;

  // We have computed the translation that maps the moving image onto the fixed
  // image, now we apply the transformation to the moving image via a resample filter	
  using ResampleFilterType = itk::ResampleImageFilter<MovingImageType, FixedImageType>;

  //  A resampling filter is created and the moving image is connected as input
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetTransform(outputBSplineTransform);
  resampler->SetInput(movingImageReader->GetOutput());

  // The resampler takes the metadata (region, spacing, origin, etc) from the fixed image)
  resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
  resampler->SetOutputOrigin(fixedImage->GetOrigin());
  resampler->SetOutputSpacing(fixedImage->GetSpacing());
  resampler->SetOutputDirection(fixedImage->GetDirection());
  resampler->SetDefaultPixelValue(100);

  // A CastImageFilter is used to convert the pixel type of the resampled image to the 
  // final type used by the writer. This would allow for processing using floating point
  // precision but taking integer valued images as input/output  
  using CastFilterType = itk::CastImageFilter<FixedImageType, FixedImageType>;
	
  // Cast and write the image
  using WriterType = itk::ImageFileWriter<FixedImageType>;
  WriterType::Pointer writer = WriterType::New();
  CastFilterType::Pointer caster = CastFilterType::New();
  writer->SetFileName(outImPath);
	
  caster->SetInput(resampler->GetOutput());
  writer->SetInput(caster->GetOutput());
  writer->Update();
  
  return EXIT_SUCCESS;
}
