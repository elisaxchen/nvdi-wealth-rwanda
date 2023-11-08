# final-project-rwanda

### Using Daytime Satellite Images, Nightlight, and NDVI to Predict Wealth: A Case Study in Rwanda

### I. Introduction

Satellite images are increasingly leveraged by social scientists to estimate economic activities in developing countries, where administrative neighborhood level data are often not reliable or simply do not exist. However, what information we'd like to extract from satellite image data can best predict wealth is contentious. The most commonly used one is nightlights satellite image, as it can capture the usage of light at night, which approximately indicates the level of development across regions. For example, in areas mired in conflict or extreme poverty, the nightlight satellite images will show more dark patches. However, this approach is less useful when it comes to estimating the variation of economic development within a low-income country. How can we find the poorest area in a country where there's not so much nightlight?

This is a critical question but there it not much written on it. Of the little there is, the paper published in Science that combines daytime satellite imagery with nightlights and survey data, exploiting CNN to identify image features to estimate consumption expenditure in Africa is a highlight (Jean et al., 2016). This project seeks to replicate and extend their main findings. In particular, I plan to investigate the possibility of using a variety of satellite images, including nightlights, daytime images, and NDVI, the vegetation index to predict wealth at the neighborhood level in Rwanda, using large-scale computing methods. 

The examination of using NDVI to estimate economy activities is of particular interest for this project, because the relationship between economic development and the preservation of greenspace is not always straightforward, especially in low-income countries like Rwanda. On the one hand, the establishment of basic urban infrastructure, a symbol of burgeoning economy, is always at the expense of the reduction of greenspace. On the other hand, social elites are more likely to preserve greenspace for themselves to have a better living environment. Can we use NDVI to predict wealth in Rwanda?


### II. Data 

For this projects, I've obtained the following data from a variety of sources: 


1. Nighttime Satellite Image Worldwide
  
Global Nighttime Lights: annual and monthly composites using VIIRS Day/Night Band 2010 is downloaded from the [NOAA (National Oceanic and Atmospheric Administration)](https://www.ncei.noaa.gov/), NCEI (National Centers for Environmental Information)'s website. This single image file provides nighttime luminosity data in the world. Combined with the Rwandan village shapefile, we'd be able to extract average nighttime luminosity for each village to assess the level of development, as a proxy for wealth. 

2. Rwandan DHS, Wealth Data
  
Demographic and Health Surveys [(DHS)](https://dhsprogram.com/methodology/Survey-Types/DHS.cfm) are nationally-representative household surveys that provide data on topics such as fertility, family planning, gender, HIV/ AIDS, malaria, and nutrition in developing countries. I've registered with the DHS, and obtained Rwandan Household File that contains household level asset information. I aggregated these households to get cluster-level wealth information. 

3. Rwanda Shapefile (Sector Boundary)

Rwanda is divided into four Provinces and the City of Kigali, containing 30 districts and 416 Sectors. The Shapefile, obtained from [Revolutionary GIS](https://github.com/justinelliotmeyers/official_rwanda_administrative_boundary_shapefile/tree/master/Sector_Boundary_2012), includes the boundary of 416 sectors in 2012. 

4. Daytime Satellite Images, Rwanda
  
I used the [Google Static Maps API](https://developers.google.com/maps/documentation/maps-static/start) to download satellite images at zoom level 16. The pixel resolution is 2.5 m. 400pixels x 2.5m = 1km, so each image covers 1 sq km. Each daytime image corresponds to a single pixel from the nighttime imagery, bounded using Rwanda shapefile.  The images are downloaded and organized into 64 folders, based on the nightlight index we calculated above.  
	
5. NDVI Image, Rwanda:
  
NDVI (Normalized Difference Vegetation Index) quantifies the density of plant growth in a pixel. It's calculated using Spectral Bands 4(Red) and  (Near Infrared - NIR) -- NIR radiation minus visible radiation divided by NIR radiation plus visible radiation. Given that plants absorb and emit solar radiation differently than urban environments, the NDVI is closer to 1 for areas covered with trees and forests, 0 for urban settings without green leaves. We obtained the NDVI image for Rwanda from [NASA's Earth Database](https://urs.earthdata.nasa.gov/oauth/authorize?client_id=ZAQpxSrQNpk342OR77kisA&response_type=code&redirect_uri=https://lpdaacsvc.cr.usgs.gov/appeears/login&state=download/9f8bb051-aa93-4d43-87a2-9e11f9b0d011): _500m_16_days_NDVI (MOD13A1.006). Note that we need to crop the image based on Rwanda shapefile.



### III. Structure of the Project

#### 1. Nighttime Satellite Images Analysis: 

This notebook contains the following: [0_Predict_Wealth_Nightlight.ipynb](https://github.com/lsc4ss-s22/final-project-rwanda/blob/ae172d51cf22b2eb6a4817f4bef0af1f2996155f/0_Predict_Wealth_Nightlight.ipynb)
- Download Nightlight Satellite Image WorldWide
- Process Rwandan DHS Wealth Data, and Combine it with Shapefile
- Extract Nightlight from the Image
- Model (Access the Relationship between Nightlight and Wealth) 

#### 2. Daytime Images Analysis:  

This notebook contains the following: [1_Predict_Wealth_Daytime_Satellite_PySpark.ipynb](https://github.com/lsc4ss-s22/final-project-rwanda/blob/ae172d51cf22b2eb6a4817f4bef0af1f2996155f/1_Predict_Wealth_Daytime_Satellite_PySpark.ipynb)
- Download Daytime Satellite Images (50,000+) with Dask
- Extract Features from Daytime Images with Dask 
- Model with PySpark Machine Learning (Predict Wealth Using RGB Features) 

#### 3. NDVI Analysis:

This notebook contains the following: [2_Predict_Wealth_NDVI_Dask.ipynb](https://github.com/lsc4ss-s22/final-project-rwanda/blob/ae172d51cf22b2eb6a4817f4bef0af1f2996155f/2_Predict_Wealth_NDVI_Dask.ipynb)
- NDVI image processing
- Extract NDVI for each village from the image with Dask
- Model (Access the Relationship between Vegetation Index and Wealth) 

### IV. Key Methods and Large-Scale Computing Techniques: 

#### 1. Nighttime Satellite Images Analysis:

- Process raster image:  
Before we process the nighttime satellite image, we first need to understand pixels. The image is a rectangular tiling of fundamental elements called pixels - an array of values with n lines and m columns. A pixel is a small block that represents the amount of color (intensity/brightness). The resolution of each pixel in the nightlight image is about 1km. 

- Extract nightlight from the image: 
Here, we use 10 pixelsx10 pixels to take the average of the luminosity values for the nightlights locations surrounding the cluster centroid. Since we only have 1 nightlight image and it’s pretty faster (less than 3 second) to process the data, we just did the serial computation.

- Here's what Rwanda looks like at night: 


![pics/night-rwanda.png](https://github.com/lsc4ss-s22/final-project-rwanda/blob/b25602ec2817c271d1249f1766b2b93012863a14/pics/night-rwanda.png)


#### 2. Daytime Images Analysis: 

- Download Daytime Satellite Images (50,000+) with Dask:
Based on the calculation above (see II.4) and each image covers 1 square km with 2.5 pixel resolution, we estimated that we’d download 50,000 + daytime images. This is a large dataset, and we decided to use Dask to parallelize the download process. We first used dask.delayed to deter execution of the function that writes images to our disk. Note that we have 53k images to download! Then we use dask.compute() to save the image, making the download process 4 times faster, compared to serial one. If we want to further scale it up, we should consider using AWS S3 to save the images. 

After downloading the pictures, we displayed the daytime images in poor / no nightlight areas vs relatively developed areas. As we can see in the images, the poor areas daytime satellite image (Left) may exhibit different patterns of RGB colors than relatively developed places (Right).  

![pics/day-img1-poor.png](https://github.com/lsc4ss-s22/final-project-rwanda/blob/b25602ec2817c271d1249f1766b2b93012863a14/pics/day-img1-poor.png)
![pics/day-img2-rich.png](https://github.com/lsc4ss-s22/final-project-rwanda/blob/b25602ec2817c271d1249f1766b2b93012863a14/pics/day-img2-rich.png)

- Process Daytime Satellite Images with Dask:
Similar to the process of extracting features from the nighttime image, we'd like to extract a set of features from daytime images that can be used to predict wealth with ml models. We first took the raw R,G,B values for each pixel and average them for the image, and then computed the min, max, median, and sd of R, G, and B for each image. This process converts each image into a vector of 15 features. Here, we'd utilize Dask again to parallelize the computation, since it can work with xarray methods [automatically](https://docs.xarray.dev/en/latest/user-guide/dask.html). Dask uses a lazy computation model that can split large arrays into smaller blocks called chunks. When we load in an image dataset as an xarray data structure, almost all xarray operations will keep it as a Dask array. And these computations, in our code, get_feature function, aren't performed until we explicitly used dask.compute() where values from a chunk are accessed. 

- Model with PySpark Machine Learning (Predict Wealth Using RGB Features):
Finally, we'd like to use daytime features to predict cluster-level wealth. Since we have relatively a large number of predictive variables, we can scale it up using PySpark with 10 folds CV and ridge regression. We first uploaded the CSV file to S3, and then used EMR PySpark to run the model. This will speed up the computing process. 

#### 3. NDVI Image Analysis:

- NDVI image processing: 
The key here is to transform the projection of NDVI imagery we obtained from NASA to specific coordinate system (EPSG, 4326) to make it consistent with Rwanda Shapefile. And then clipped the satellite image (NDVI) with Rwanda shapefile. This is what we got from the original tif file to Rwanda NDVI image: 


![pics/ndvi-original](https://github.com/lsc4ss-s22/final-project-rwanda/blob/ae172d51cf22b2eb6a4817f4bef0af1f2996155f/pics/ndvi-original.png)


![pics/ndvi-rwanda.png](https://github.com/lsc4ss-s22/final-project-rwanda/blob/ae172d51cf22b2eb6a4817f4bef0af1f2996155f/pics/ndvi-rwanda.png)

- Extract NDVI for each village from the image with Dask:
Since we need to repeatedly clip NDVI images file with Rwanda shapefile mapping out 416 villages, and then calculate the mean NDVI for each village, we'd utilize Dask again to delay the execution of operation. The process is similar to extracting features with Dask, described above. 

After calculating the NDVI for each village, we notice that there exist a discrepancy between village shapefiles and cluster wealth. We have 416 cluster  geometries, but 492 cluster-level wealth data. To solve this issue, we defined a function to match a specific cluster, based on its coordination, longitude and latitude, to a existing geometry boundary. 

### V. Findings

#### a. Nighttime Images 

![pics/night-plot.png](https://github.com/lsc4ss-s22/final-project-rwanda/blob/281a80d8218d129bc4bca99750baf0b550094ebe/pics/night-plot.png)
![pics/night-reg.png](https://github.com/lsc4ss-s22/final-project-rwanda/blob/281a80d8218d129bc4bca99750baf0b550094ebe/pics/night-reg.png)

As we can see in the plot above, there's strong positive relationship between nightlight and wealth level. The result is statistically siginificant. However, we can also see that most villeges do not have any light at all during night. It's hard to differernciate the poorest village from a cluster of poor villages! Note that we calculate the average light among 10x10 pixels. It might worth trying to extract min, median, and standard deviation of the nightlight to add more features, and then use 10 folds CV to predict wealth. The nightlight can be a good proxy for wealth level, but as expected, it's hard to make predictions when most clusters have no light. 

#### b. Daytime Images 

![pics/day-r2.png](https://github.com/lsc4ss-s22/final-project-rwanda/blob/281a80d8218d129bc4bca99750baf0b550094ebe/pics/day-r2.png) 

We built the regression model in EMR PySpark, using 10 folds CV and ridge regression. Here, we have 15 features, including min, max, median, mean, and standard deviations of R, G, B for each 10x10 pixels. The average R^2 is around 0.35, which means this model can explain around 35% of variation in the wealth level. Not too bad. 

#### c. NDVI 

![pics/ndvi-wealth-plot.png](https://github.com/lsc4ss-s22/final-project-rwanda/blob/281a80d8218d129bc4bca99750baf0b550094ebe/pics/ndvi-wealth-plot.png)
![pics/ndvi-reg.png](https://github.com/lsc4ss-s22/final-project-rwanda/blob/281a80d8218d129bc4bca99750baf0b550094ebe/pics/ndvi-reg.png)

Finally, we'd like to explore the relationship between NDVI and wealth level. There's a slighly negative, but statiscally significant relationship between NDVI, the vegetation index, and cluster wealth. This shows that relatively developed rich areas have less greenspace in Rwanda.

Overall, we can see that RGB from daytime satellite images, nightlight, and NDVI can all serve as predictors of cluster wealth. This project doesn't dig into more advanced ML methods like the original paper, such as CNN to extract more features from images due to time limit, but it provides the possibility of using NDVI to add more features to the model. If we'd like to have granular data of wealth on a smaller geography, we could use a smaller division, within the 416 sectors in Rwanda, and process the images accordingly. I didn't find the shapefile for these divisions though. 

### VI. References 

The replication process followes the workflow of this notebook: https://github.com/jblumenstock/bdd/blob/master/PS1/BDD2021-PS1.ipynb. It also provides some helpful functions to get started (ie, read a raster file, get the pixel index of the point, download images from google API). In my code, I modified them and scaled it up, using dask, as I see fit.  

For extracting features (RGB) from daytime images, I've consulted this repo: https://github.com/nealjean/predicting-poverty; and https://towardsdatascience.com/satellite-imagery-analysis-using-python-9f389569862c

For extracting NDVI information from NDVI images, I've consulted this notebook: https://github.com/MengChenC/NDVI-and-Human-Well-being/blob/main/Jupyter%20Notebooks/Gather_Monthly_Vegetation_with_Dask.ipynb

For clipping raster file with a shapefile & tranforming the projection of imagery to specific coordinate system, I've consulted this notebook https://thinkinfi.com/clip-raster-with-a-shape-file-in-python/


