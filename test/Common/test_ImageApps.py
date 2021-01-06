# -*- coding: utf-8 -*-

import unittest
from Common.GDalDatasetWrapper import GDalDatasetWrapper
from Common import ImageApps
import numpy as np


class TestImageTools(unittest.TestCase):

    def setUp(self):
        self.coordinates = (652594.9112913811, 10.00887639510383, 0,
                            5072876.717295351, 0, -9.974893672262251)
        self.projection = 'PROJCS["WGS 84 / UTM zone 31N",\
                           GEOGCS["WGS 84",DATUM["WGS_1984",\
                           SPHEROID["WGS 84",6378137,298.257223563,\
                           AUTHORITY["EPSG","7030"]],\
                           AUTHORITY["EPSG","6326"]],\
                           PRIMEM["Greenwich",0,\
                           AUTHORITY["EPSG","8901"]],\
                           UNIT["degree",0.0174532925199433,\
                           AUTHORITY["EPSG","9122"]],\
                           AUTHORITY["EPSG","4326"]],\
                           PROJECTION["Transverse_Mercator"],\
                           PARAMETER["latitude_of_origin",0],\
                           PARAMETER["central_meridian",3],\
                           PARAMETER["scale_factor",0.9996],\
                           PARAMETER["false_easting",500000],\
                           PARAMETER["false_northing",0],\
                           UNIT["metre",1,AUTHORITY["EPSG","9001"]],\
                           AXIS["Easting",EAST],AXIS["Northing",NORTH],\
                           AUTHORITY["EPSG","32631"]]'
        self.height = 200
        self.width = 100

    def test_get_ndsi_float32(self):
        swir = np.arange(0, 9).reshape(3, 3)
        red = np.ones((3, 3))
        swir[-1, -1] = 0
        red[-1, -1] = 0
        ds_swir = GDalDatasetWrapper(array=swir, projection=self.projection, geotransform=self.coordinates)
        ds_red = GDalDatasetWrapper(array=red, projection=self.projection, geotransform=self.coordinates)
        expected = np.array([[1, 0, -.3333333],
                            [-.5, -.6, -.6666667],
                            [-0.7142857, -0.75, -1]], dtype=np.float32)

        calculated = ImageApps.get_ndsi(ds_red, ds_swir)
        np.testing.assert_almost_equal(calculated.array, expected)
        self.assertEqual(calculated.resolution, ds_swir.resolution)
        self.assertEqual(calculated.resolution, ds_red.resolution)
        self.assertEqual(calculated.extent, ds_red.extent)
        self.assertEqual(calculated.extent, ds_swir.extent)

    def test_get_ndsi_int16(self):
        swir = np.array(np.arange(0, 9).reshape(3, 3), dtype=np.int16)
        red = np.ones((3, 3), dtype=np.int16)
        swir[-1, -1] = 0
        red[-1, -1] = 0
        ds_swir = GDalDatasetWrapper(array=swir, projection=self.projection, geotransform=self.coordinates)
        ds_red = GDalDatasetWrapper(array=red, projection=self.projection, geotransform=self.coordinates)

        expected = np.array([[1000, 500, 333],
                            [250, 199, 166],
                            [142, 125, 0]], dtype=np.int16)

        calculated = ImageApps.get_ndsi(ds_red, ds_swir, vrange=(0, 1000), dtype=np.int16)
        np.testing.assert_almost_equal(calculated.array, expected)
        self.assertEqual(calculated.resolution, ds_swir.resolution)
        self.assertEqual(calculated.resolution, ds_red.resolution)
        self.assertEqual(calculated.extent, ds_red.extent)
        self.assertEqual(calculated.extent, ds_swir.extent)

    def test_get_ndvi_float32(self):
        nir = np.arange(0, 9).reshape(3, 3)
        red = np.ones((3, 3))
        nir[-1, -1] = 0
        red[-1, -1] = 0

        ds_nir = GDalDatasetWrapper(array=nir, projection=self.projection, geotransform=self.coordinates)
        ds_red = GDalDatasetWrapper(array=red, projection=self.projection, geotransform=self.coordinates)

        expected = np.array([[1, 0, -.3333333],
                            [-.5, -.6, -.6666667],
                            [-0.7142857, -0.75, -1]], dtype=np.float32)

        calculated = ImageApps.get_ndvi(ds_red, ds_nir)
        np.testing.assert_almost_equal(calculated.array, expected)
        self.assertEqual(calculated.resolution, ds_nir.resolution)
        self.assertEqual(calculated.resolution, ds_red.resolution)
        self.assertEqual(calculated.extent, ds_red.extent)
        self.assertEqual(calculated.extent, ds_nir.extent)

    def test_get_ndvi_int16(self):
        nir = np.arange(0, 9).reshape(3, 3)
        red = np.ones((3, 3))
        ds_nir = GDalDatasetWrapper(array=nir, projection=self.projection, geotransform=self.coordinates)
        ds_red = GDalDatasetWrapper(array=red, projection=self.projection, geotransform=self.coordinates)
        nir[-1, -1] = 0
        red[-1, -1] = 0

        expected = np.array([[1000, 500, 333],
                            [250, 199, 166],
                            [142, 125, 0]], dtype=np.int16)

        calculated = ImageApps.get_ndvi(ds_red, ds_nir, vrange=(0, 1000), dtype=np.int16)
        np.testing.assert_almost_equal(calculated.array, expected)
        self.assertEqual(calculated.resolution, ds_nir.resolution)
        self.assertEqual(calculated.resolution, ds_red.resolution)
        self.assertEqual(calculated.extent, ds_red.extent)
        self.assertEqual(calculated.extent, ds_nir.extent)

    def test_split_image(self):
        stacked = np.arange(0, 12).reshape((2, 2, 3))
        vstack = stacked.reshape(-1, 3)
        self.assertEqual(vstack.shape, (4, 3))

        splitted = np.array_split(vstack, 3, axis=0)
        # Extract single band:
        splitted_b0 = [s[:, 0] for s in splitted]

        # Unravel
        unraveled = np.concatenate(splitted_b0).reshape((2, 2))
        np.testing.assert_almost_equal(unraveled, stacked[..., 0])


if __name__ == '__main__':
    unittest.main()
