import sys
import os
from osgeo import gdal, osr, ogr
import numpy as np
import json
import numpy as np
from PIL import Image


def create_poly_mask(rasterSrc, vectorSrc, npDistFileName='', noDataValue=0, burn_values=255):

	'''
	Create polygon mask for rasterSrc,
	Similar to labeltools/createNPPixArray() in spacenet utilities
	'''

	## open source vector file that truth data
	source_ds = ogr.Open(vectorSrc)
	source_layer = source_ds.GetLayer()

	## extract data from src Raster File to be emulated
	## open raster file that is to be emulated
	srcRas_ds = gdal.Open(rasterSrc)
	cols = srcRas_ds.RasterXSize
	rows = srcRas_ds.RasterYSize

	if npDistFileName == '':
		dstPath = ".tmp.tiff"
	else:
		dstPath = npDistFileName

	## create First raster memory layer, units are pixels
	# Change output to geotiff instead of memory 
	memdrv = gdal.GetDriverByName('GTiff') 
	dst_ds = memdrv.Create(dstPath, cols, rows, 1, gdal.GDT_Byte, 
							options=['COMPRESS=LZW'])
	dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
	dst_ds.SetProjection(srcRas_ds.GetProjection())
	band = dst_ds.GetRasterBand(1)
	band.SetNoDataValue(noDataValue)    
	gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
	dst_ds = 0
	mask_image = Image.open(dstPath)
	mask_image = np.array(mask_image)

	if npDistFileName == '':
		os.remove(dstPath)
		
	return mask_image


def geojson_to_pixel_arr(raster_file, geojson_file, pixel_ints=True, verbose=False):
	'''
	Tranform geojson file into array of points in pixel (and latlon) coords
	pixel_ints = 1 sets pixel coords as integers
	'''
	
	# load geojson file
	with open(geojson_file) as f:
		geojson_data = json.load(f)

	# load raster file and get geo transforms
	src_raster = gdal.Open(raster_file)
	targetsr = osr.SpatialReference()
	targetsr.ImportFromWkt(src_raster.GetProjectionRef())
		
	geom_transform = src_raster.GetGeoTransform()
	
	# get latlon coords
	latlons = []
	types = []
	for feature in geojson_data['features']:
		coords_tmp = feature['geometry']['coordinates'][0]
		type_tmp = feature['geometry']['type']
		if verbose: 
			print("features:", feature.keys())
			print("geometry:features:", feature['geometry'].keys())

			#print("feature['geometry']['coordinates'][0]", z)
		latlons.append(coords_tmp)
		types.append(type_tmp)
		#print(feature['geometry']['type'])
	
	# convert latlons to pixel coords
	pixel_coords = []
	latlon_coords = []
	for i, (poly_type, poly0) in enumerate(zip(types, latlons)):
		
		if poly_type.upper() == 'MULTIPOLYGON':
			#print("oops, multipolygon")
			for poly in poly0:
				poly=np.array(poly)
				if verbose:
					print("poly.shape:", poly.shape)
					
				# account for nested arrays
				if len(poly.shape) == 3 and poly.shape[0] == 1:
					poly = poly[0]
					
				poly_list_pix = []
				poly_list_latlon = []
				if verbose: 
					print("poly", poly)
				for coord in poly:
					if verbose: 
						print("coord:", coord)
					lon, lat = coord 
					px, py = latlon2pixel(lat, lon, input_raster=src_raster,
										 targetsr=targetsr, 
										 geom_transform=geom_transform)
					poly_list_pix.append([px, py])
					if verbose:
						print("px, py", px, py)
					poly_list_latlon.append([lat, lon])
				
				if pixel_ints:
					ptmp = np.rint(poly_list_pix).astype(int)
				else:
					ptmp = poly_list_pix
				pixel_coords.append(ptmp)
				latlon_coords.append(poly_list_latlon)            

		elif poly_type.upper() == 'POLYGON':
			poly=np.array(poly0)
			if verbose:
				print("poly.shape:", poly.shape)
				
			# account for nested arrays
			if len(poly.shape) == 3 and poly.shape[0] == 1:
				poly = poly[0]
				
			poly_list_pix = []
			poly_list_latlon = []
			if verbose: 
				print("poly", poly)
			for coord in poly:
				if verbose: 
					print("coord:", coord)
				lon, lat = coord 
				px, py = latlon2pixel(lat, lon, input_raster=src_raster,
									 targetsr=targetsr, 
									 geom_transform=geom_transform)
				poly_list_pix.append([px, py])
				if verbose:
					print("px, py", px, py)
				poly_list_latlon.append([lat, lon])
			
			if pixel_ints:
				ptmp = np.rint(poly_list_pix).astype(int)
			else:
				ptmp = poly_list_pix
			pixel_coords.append(ptmp)
			latlon_coords.append(poly_list_latlon)
			
		else:
			print("Unknown shape type in coords_arr_from_geojson()")
			return
			
	return pixel_coords, latlon_coords


def latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):

	sourcesr = osr.SpatialReference()
	sourcesr.ImportFromEPSG(3857)

	geom = ogr.Geometry(ogr.wkbPoint)
	geom.AddPoint(lon, lat)

	if targetsr == '':
		src_raster = gdal.Open(input_raster)
		targetsr = osr.SpatialReference()
		targetsr.ImportFromWkt(src_raster.GetProjectionRef())
	coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
	if geom_transform == '':
		src_raster = gdal.Open(input_raster)
		transform = src_raster.GetGeoTransform()
	else:
		transform = geom_transform

	x_origin = transform[0]
	# print(x_origin)
	y_origin = transform[3]
	# print(y_origin)
	pixel_width = transform[1]
	# print(pixel_width)
	pixel_height = transform[5]
	# print(pixel_height)
	geom.Transform(coord_trans)
	# print(geom.GetPoint())
	x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
	y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

	return (x_pix, y_pix)