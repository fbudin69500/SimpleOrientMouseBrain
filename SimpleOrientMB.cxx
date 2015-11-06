#include "SimpleOrientMBCLP.h"
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageMomentsCalculator.h>
#include "itkImageRegionConstIteratorWithIndex.h"
#include <vnl/vnl_math.h>
#include <limits>
#include <itkAffineTransform.h>
#include <itkMatrix.h>
#include <itkTransformFileWriter.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

double ComputeSymmetryCoefficient( itk::Image< float , 3 >::Pointer image ,
                                   itk::ContinuousIndex< double , 3 >   ContIndex ,
                                   int pos ,
                                   int dir ,
                                   unsigned int radius
                                 )
{
  typedef itk::Image< float , 3 > ImageType ;
  double coeff = 0 ;
  typedef itk::ImageRegionConstIteratorWithIndex< ImageType > IteratorType ;
  ImageType::RegionType region ;
  ImageType::SizeType size ;
  size = image->GetLargestPossibleRegion().GetSize() ;
  if( size[ dir ] < 2 )
  {
    return -1 ;
  }
  ImageType::IndexType index ;
  ImageType::SizeType regionSize ;
  //find 3rd direction
  int third = -1 ;
  for( int i = 0 ; i < 3 ; i++ )
  {
    if( i != pos && i != dir )
    {
      third = i ;
      break ;
    }
  }
  if( third < 0 )
  {
    return -1 ;
  }
  if( vnl_math_rnd( ContIndex[ third ] ) < signed(radius) )
  {
    radius = vnl_math_rnd( ContIndex[ third ] ) ;
  }
  if( signed(size[ third ]) <= signed(vnl_math_rnd( ContIndex[ third ] ) + radius) )
  {
    radius = size[ third ] - vnl_math_rnd( ContIndex[ third ] ) - 1 ;
  }
//  std::cout<< size[ third ]<<" " <<vnl_math_rnd( ContIndex[ third ] )<<" " << radius << std::endl ;
//  std::cout<<"dir: "<<dir << std::endl;
//  std::cout<<"third: "<<third << std::endl;
  index[ third ] = vnl_math_rnd( ContIndex[ third ] ) - radius ;
  index[ pos ] = 0 ;
  regionSize[ third ] = 2 * radius + 1 ;
  regionSize[ pos ] = size[ pos ] ;
  //we find which side is the smallest
//  std::cout<<"contindex:"<<ContIndex<<std::endl;
  if( signed(vnl_math_floor( ContIndex[ dir ] ) + 1) < signed(size[ dir ] - vnl_math_ceil( ContIndex[ dir ] ) ) )
  {
    index[ dir ] = 0 ;
    regionSize[ dir ] = vnl_math_floor( ContIndex[ dir ] ) + 1 ;
  }
  else
  {
    index[ dir ] = vnl_math_ceil( ContIndex[ dir ] ) ;
    regionSize[ dir ] = size[ dir ] - vnl_math_ceil( ContIndex[ dir ] ) ;
  }
  long sizeDir = vnl_math_floor( ContIndex[ dir ] ) * 2 + 1 ;//symmetry
//  std::cout<<"sizeDir: "<<sizeDir<<std::endl;
  region.SetSize( regionSize ) ;
  region.SetIndex( index ) ;
//  std::cout<<"regionSize:"<<regionSize<<std::endl;
//  std::cout<<"index:"<<index<<std::endl;
  IteratorType it( image, region ) ;
  for( it.GoToBegin() ; !it.IsAtEnd() ; ++it )
  {
    double temp , temp2 ;
    temp = it.Get() ;
    index = it.GetIndex() ;
//    std::cout<<"index:"<<index<<std::endl;
    index[ dir ] = sizeDir - index[ dir ] ;
//    std::cout<<"symindex:"<<index<<std::endl;
    temp2 = image->GetPixel( index ) ;
    temp -= temp2 ;
    temp = ( temp > 0 ? temp : -temp ) ;
    coeff += temp ;
  }
  return coeff ;
}


void SetRegion( itk::Image< float , 3 >::Pointer image ,
                float indexPos ,
                int pos ,
                bool after
              )
{
  typedef itk::Image< float , 3 > ImageType ;
  ImageType::RegionType region ;
  ImageType::SizeType size ;
  ImageType::IndexType regionIndex ;
  size = image->GetLargestPossibleRegion().GetSize() ;
  regionIndex.Fill( 0 ) ;
  if( !after )
  {
    size[ pos ] = vnl_math_floor( indexPos ) + 1 ;
  }
  else
  {
    size[ pos ] = size[ pos ] - vnl_math_ceil( indexPos ) ;
    regionIndex[ pos ] = vnl_math_ceil( indexPos ) ;
  }
  region.SetIndex( regionIndex ) ;
  region.SetSize( size ) ;
  image->SetRequestedRegion( region ) ;
}


double ComputeDistanceToCenterLine( itk::Image<float , 3 >::Pointer image ,
                                    itk::Point< double , 3 > axis ,
                                    int pos
                                  )
{
  typedef itk::Image< float , 3 > ImageType ;
  typedef itk::ImageRegionConstIteratorWithIndex< ImageType > IteratorType ;
  IteratorType it( image , image->GetRequestedRegion() ) ;
  double sum ;
  double sumMult ;
  itk::Index< 3 > index ;
  itk::Point< double , 3 > point ;
  for( it.GoToBegin() ; !it.IsAtEnd() ; ++it )
  {
    double value = it.Get() ;
    sum += value ;
    index = it.GetIndex() ;
    image->TransformIndexToPhysicalPoint( index , point ) ;
    axis[ pos ] = point[ pos ] ;
    double dist = point.EuclideanDistanceTo( axis ) ;
    sumMult += value * dist ;
  }
  return sumMult/sum ;
}

int main( int argc, char * argv[] )
{
  PARSE_ARGS;
  typedef itk::Image< float , 3 > ImageType ;
  itk::ImageFileReader< ImageType >::Pointer reader = itk::ImageFileReader< ImageType >::New() ;
  reader->SetFileName( inputVolume.c_str() ) ;
  reader->Update() ;
//  std::cout<<reader->GetOutput()->GetLargestPossibleRegion().GetSize() <<std::endl;
  typedef itk::ImageMomentsCalculator< ImageType > MomentFilterType ;
  MomentFilterType::Pointer MomentFilter = MomentFilterType::New() ;
  MomentFilter->SetImage( reader->GetOutput() ) ;
  MomentFilter->Compute() ;
  MomentFilterType::MatrixType principalAxes = MomentFilter->GetPrincipalAxes();
  MomentFilterType::VectorType centerOfGravity = MomentFilter->GetCenterOfGravity() ;
  //compute along which axis the projection of the principal axis with the largest principal moment
  itk::Vector< double , 3 > axis ;
  itk::Point< double , 3 > center ;
  std::vector< itk::Vector< double , 3 > > frame( 3 ) ;
  int i ;
  for( int i = 0 ; i < 3 ; i++ )
  {
    axis[ i ] = principalAxes[ 2 ][ i ] ;
    center[ i ] = centerOfGravity[ i ] ;
    for( int j = 0 ; j < 3 ; j++ )
    {
      frame[ i ][ j ] = (i==j) ;
    }
  }
  double val = -1.0 ;
  int pos = -1 ;
  for( i = 0 ; i < 3 ; i++ )
  {
    double temp ;
    temp = dot_product( axis.GetVnlVector() , frame[ i ].GetVnlVector() ) ;
    temp = ( temp >= 0 ? temp: -temp ) ;
    if( temp > val )
    {
      val = temp ;
      pos = i ;
    }
  }
  if( pos == -1 )
  {
    std::cerr << "Error: No longest axis!!!!" << std::endl ;
    return EXIT_FAILURE ;
  }
//  std::cout << pos << std::endl ;
  //Compute symmetry
  itk::ContinuousIndex< double , 3 > index ;
  reader->GetOutput()->TransformPhysicalPointToContinuousIndex( center , index ) ;
  double coeff = std::numeric_limits< double >::max() ;
  int pos2 = -1 ;
  for( i = 0 ; i < 3 ; i++ )
  {
    if( i != pos )
    {
      double temp ;
      temp = ComputeSymmetryCoefficient( reader->GetOutput() , index , pos , i , radius ) ;
//      std::cout << temp << std::endl ;
      if( temp < 0 )
      {
        std::cerr << "Cannot compute symmetry along this direction: image size < 1" << std::endl ;
        continue ;
      }
      if( temp < coeff )
      {
        coeff = temp ;
        pos2 = i ;
      }
    }
  }
  if( pos2 == -1 )
  {
    std::cerr << "Error: No axis of symmetry!!!!" << std::endl ;
    return EXIT_FAILURE ;
  }
  int third = -1 ;
  for( int i = 0 ; i < 3 ; i++ )
  {
    if( i != pos && i != pos2 )
    {
      third = i ;
      break ;
    }
  }
//Compute front and back
SetRegion( reader->GetOutput() , index[ pos ] , pos , 0 ) ;
double dist1 = ComputeDistanceToCenterLine( reader->GetOutput() , center , pos ) ;
std::cout << dist1 << std::endl ;
SetRegion( reader->GetOutput() , index[ pos ] , pos , 1 ) ;
double dist2 = ComputeDistanceToCenterLine( reader->GetOutput() , center , pos ) ;
std::cout << dist2 << std::endl ;
///Print results and save transform
  std::cout << "longest:" << pos << std::endl ;
  std::cout << "symmetry:" << pos2 << std::endl ;
  itk::Matrix< double , 3 , 3 > matrix ;
  for( i = 0 ; i < 3 ; i++ )
  {
    matrix[ i ][ 0 ] = frame[ i ][ pos2 ] ;
    matrix[ i ][ 1 ] = frame[ i ][ pos ] ;
    matrix[ i ][ 2 ] = frame[ i ][ third ] ;
  }
  typedef itk::AffineTransform< double , 3 > AffineType ;
  AffineType::Pointer affine = AffineType::New() ;
  affine->SetMatrix( matrix ) ;
  typedef itk::TransformFileWriter WriterType ;
  WriterType::Pointer writer = WriterType::New() ;
  writer->SetFileName( transform ) ;
  writer->AddTransform( affine ) ;
  writer->Update() ;
  return EXIT_SUCCESS ;
}
