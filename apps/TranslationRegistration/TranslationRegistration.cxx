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
#include "itkCastImageFilter.h"
#include "itkImage.h"
#include "itkImageRegistrationMethod.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSpatialObjectToImageFilter.h"
#include "itkTranslationTransform.h"

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
	  
	  using OptimizerType = itk::RegularStepGradientDescentOptimizer;
	  using OptimizerPointer = const OptimizerType *;
	
	  void Execute(itk::Object * caller, const itk::EventObject & event) override
	  {
      Execute((const itk::Object *)caller, event);
	  }
	
	  void Execute(const itk::Object * object, const itk::EventObject & event) override
	  {
      auto optimizer = static_cast<OptimizerPointer>(object);
	
      if (!itk::IterationEvent().CheckEvent(&event))
      {
        return;
      }
      
      // Print iteration number and current value of metric
      std::cout << "  Iteration " << optimizer->GetCurrentIteration()+1 << ": " << optimizer->GetValue() << "\n";
	  }
};

int main(int argc, char ** argv)
{
  
  // If there are too few arguments print a usage message
  if (argc < 4)
  {
    std::cout << "USAGE: TranslationRegistration fixed_image moving_image output_image\n";
    return EXIT_FAILURE;
  }
  
  // The image type for this application is 3 dimensional with floating point values 
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, Dimension>;
  
  // Read command line arguments
  char* fixedImPath = argv[1];      // Path to the fixed image
  char* movingImPath = argv[2];     // Path to the moving image
  char* outImPath = argv[3];        // Desired output path and file name
  
  // Read the fixed and moving images
  using ReaderType = itk::ImageFileReader<ImageType>;
  
  ReaderType::Pointer fixedReader = ReaderType::New();
  ReaderType::Pointer movingReader = ReaderType::New();
  
  fixedReader->SetFileName(fixedImPath);
  movingReader->SetFileName(movingImPath);
  
  fixedReader->Update();
  movingReader->Update();
  
  // Store the fixed and moving images
  ImageType::Pointer fixedImage = fixedReader->GetOutput();
  ImageType::Pointer movingImage = movingReader->GetOutput();

  // The transform that will map the fixed image into the moving image.
  // In this case it is a 3D translation (x,y,z)
  using TransformType = itk::TranslationTransform<double, Dimension>;
	
  // An optimizer is required to explore the parameter space of the transform
  // in search of optimal values of the metric
  using OptimizerType = itk::RegularStepGradientDescentOptimizer;

  // The metric will compare how well the two images match each other. Metric
  // types are usually parameterized by the image types as it can be seen in
  // the following type declaration. Mean squares is a direct comparison
  // between intensity values.
  using MetricType = itk::MeanSquaresImageToImageMetric<ImageType, ImageType>;
	
  // Finally, the type of the interpolator is declared. The interpolator will
  // evaluate the intensities of the moving image at non-grid positions
  using InterpolatorType = itk::LinearInterpolateImageFunction<ImageType, double>;
	
  // The registration method type is instantiated using the types of the
  // fixed and moving images. This class is responsible for interconnecting
  // all the components that we have described so far.
  using RegistrationType = itk::ImageRegistrationMethod<ImageType, ImageType>;
	
  // Create components
  MetricType::Pointer metric = MetricType::New();
  TransformType::Pointer transform = TransformType::New();
  OptimizerType::Pointer optimizer = OptimizerType::New();
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  RegistrationType::Pointer registration = RegistrationType::New();
	
  // Each component is now connected to the instance of the registration method
  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetTransform(transform);
  registration->SetInterpolator(interpolator);
	
  // Set the registration inputs
  registration->SetFixedImage(fixedImage);
  registration->SetMovingImage(movingImage);
	registration->SetFixedImageRegion(fixedImage->GetLargestPossibleRegion());
	
  // Initialize the transform parameters with no translation (0,0,0)
  using ParametersType = RegistrationType::ParametersType;
  ParametersType initialParameters(transform->GetNumberOfParameters());
	
  initialParameters[0] = 0.0; // Initial offset along X
  initialParameters[1] = 0.0; // Initial offset along Y
  initialParameters[2] = 0.0; // Initial offset along Z
	
  registration->SetInitialTransformParameters(initialParameters);
	
  // Optimizer settings
  optimizer->SetMaximumStepLength(4.00);
  optimizer->SetMinimumStepLength(0.01);
	optimizer->SetNumberOfIterations(200);
  
  // For updating the user on the status of the optimization
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver(itk::IterationEvent(), observer);
	
  std::cout << "\nBeginning registration...\n\n";
  
  // Start the registration optimization
  try
  {
	  registration->Update();
  }
  catch (itk::ExceptionObject & err)
  {
	  std::cerr << "ExceptionObject caught !" << std::endl;
	  std::cerr << err << std::endl;
	  return EXIT_FAILURE;
  }
	
  // Grab the transformation parameters at convergence
	ParametersType finalParameters = registration->GetLastTransformParameters();
	
  const double TranslationAlongX = finalParameters[0];
  const double TranslationAlongY = finalParameters[1];
  const double TranslationAlongZ = finalParameters[2];
	
  // Grab the final number of iterations it took to reach convergence	
  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
	
  // Grab the final value of the metric	
  const double bestValue = optimizer->GetValue();
	
  // Print out results
  std::cout << "\nResult = " << std::endl;
  std::cout << "  Translation X = " << TranslationAlongX << std::endl;
  std::cout << "  Translation Y = " << TranslationAlongY << std::endl;
  std::cout << "  Translation Z = " << TranslationAlongZ << std::endl;
  std::cout << "  Iterations = " << numberOfIterations << std::endl;
  std::cout << "  Metric value = " << bestValue << std::endl;
	
  // We have computed the translation that maps the moving image onto the fixed
  // image, now we apply the transformation to the moving image via a resample filter	
  using ResampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
	
  //  A resampling filter is created and the moving image is connected as input
	ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetInput(movingImage);
	resampler->SetTransform(registration->GetOutput()->Get());
	
  // The resampler takes the metadata (region, spacing, origin, etc) from the fixed image)	
  resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
  resampler->SetOutputOrigin(fixedImage->GetOrigin());
  resampler->SetOutputSpacing(fixedImage->GetSpacing());
  resampler->SetOutputDirection(fixedImage->GetDirection());
  resampler->SetDefaultPixelValue(100);
	
  // A CastImageFilter is used to convert the pixel type of the resampled image to the 
  // final type used by the writer. This would allow for processing using floating point
  // precision but taking integer valued images as input/output  
  using CastFilterType = itk::CastImageFilter<ImageType, ImageType>;
	
  // Cast and write the image
  using WriterType = itk::ImageFileWriter<ImageType>;
  WriterType::Pointer writer = WriterType::New();
  CastFilterType::Pointer caster = CastFilterType::New();
  writer->SetFileName(outImPath);
	
  caster->SetInput(resampler->GetOutput());
  writer->SetInput(caster->GetOutput());
  writer->Update();
  
  return EXIT_SUCCESS;

}
