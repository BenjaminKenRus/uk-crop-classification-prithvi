# The Dataset is Based on the IBM-NASA Temporal Crop Classification Dataset Guidelines

- [IBM-NASA Guidlines](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification)
- [IBM-NASA Dataset Card](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)
- [Clark Center for Geospatial Analytics Dataset](https://source.coop/clarkcga/multi-temporal-crop-classification)
 
## TIFF File Description
The dataset includes ~900 raster GeoTIFF input chips (images and labels). Image chips are extracted from Sentinel-2. Each image chip contains 18 bands including 6 spectral bands for three time-steps stacked together. Label chips are from [UKCEH Land Cover® plus: Crops](https://www.ceh.ac.uk/data/ceh-land-cover-plus-crops-2015) classified into 16 classes containing one band with the target classes for each pixel.
 
### Band Order
In each input GeoTIFF the following bands are repeated three times for three observations throughout the growing season:

- Blue, (RGB) 
- Green, (RGB) 
- Red, (RGB) 
- NIR, (Narrow NIR) 
- SW 1, (SWIR 1) 
- SW 2, (SWIR 2)

