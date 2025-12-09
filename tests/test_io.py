import unittest
import os
import BlackTelperion
from BlackTelperion import io
from pathlib import Path
from tempfile import mkdtemp
import shutil
import numpy as np

#TODO:
# 1. Test Black collections and BlackTelperion.analyze parts
# 2. Test with real images from different platforms

class TestIO(unittest.TestCase):
    def test_load(self):

        if io.usegdal:
            test = [False, True] # test both GDAL and SPy
        else:
            test = [False] # only test SPy - no gdal
        for gdal in test:
            io.usegdal = gdal
            self.img = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
            self.lib = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"library.csv"))

            # test load with SPy
            img = io.loadWithSPy(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
            self.assertTrue( np.nanmax( np.abs(self.img.data - img.data ) ) < 0.01 ) # more or less equal datsets

    def test_loadtxt(self):
        lib = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "library.csv"))
        pth = mkdtemp()
        # BlackLibrary.data =>  a numpy array containing spectral measurements and indexed as
        # [sample, measurement, band]. If different numbers of measurements are available
        # per sample then some of the measurement axes can be set as nan.
        # If a 2D array is passed (sample, band) then it will be expanded to (sample, 1, band).
        self.assertEqual(3, len(lib.data.shape)) # test data shape
        self.assertEqual(3, lib.data.shape[0]) # test data samples
        self.assertEqual(450, lib.data.shape[2]) # test wavelength count

        try:
            io.saveLibraryTXT(os.path.join(pth,"libtxt.txt"), lib )
            io.saveLibraryCSV(os.path.join(pth, "libcsv.csv"), lib)

            lib2 = io.loadLibraryTXT(os.path.join(pth,"libtxt.txt"))
            lib3 = io.loadLibraryCSV(os.path.join(pth, "libcsv.csv"))
            for l in [lib2, lib3]:
                self.assertLess( np.max( np.abs( l.data - lib.data ) ), 1e-5 )
                self.assertLess( np.max(np.abs(l.get_wavelengths() - lib.get_wavelengths())), 1e-5 )

            # test loading from directory
            for i,mineral in enumerate(['quartz', 'biotite','phlogopite']): # build directory
                io.saveLibraryTXT(os.path.join(pth,"library/%s/_%d.txt"%(mineral,i)), lib )
            lib = io.loadLibraryDIR(os.path.join(pth,"library"))
            self.assertIn('phlogopite', lib.get_sample_names())

        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not load or save spectral library to text format.")

    def test_save(self):
        self.test_load() # load datasets
        pth = mkdtemp()

        if io.usegdal:
            test = [False, True] # test both GDAL and SPy
        else:
            print("Warning - GDAL is not installed. GDAL related functions will not be tested.")
            test = [False] # only test SPy - no gdal

        for gdal in test:
            io.usegdal = gdal
            # test lib and cloud
            try:
                data = self.lib
                name = "lib"
                path = os.path.join(pth, f"{name}.hdr")  # or "%s.hdr" % name

                io.save(path, data)
                data2 = io.load(path)

                self.assertAlmostEqual(np.nanmax(np.abs(data.data - data2.data)),0.0,places=6)  # check values are the same after saving

                # test image(s) with GDAL and SPy
                for data in [self.img]:
                        # save with default (GDAL?)
                        io.save(os.path.join(pth, "data.hdr"), data )
                        self.assertEqual( os.path.exists(os.path.join(pth, "data.hdr")), True)
                        data2 = io.load(os.path.join(pth, "data.hdr")) # reload it
                        self.assertAlmostEqual(np.nanmax(np.abs(data.data - data2.data)), 0,
                                                6)  # check values are the same

                        # save with SPy
                        io.saveWithSPy(os.path.join(pth, "data2.hdr"), data )
                        self.assertEqual(os.path.exists(os.path.join(pth, "data2.hdr")), True)
                        data2 = io.load(os.path.join(pth, "data2.hdr")) # reload it
                        self.assertAlmostEqual(np.nanmax(np.abs(data.data - data2.data)), 0,
                                                6)  # check values are the same

                # test saving 3-band images to png files
                rgb = self.img.export_bands(BlackTelperion.RGB) # get 3-band image
                rgb.percent_clip(1,99,per_band=True,clip=True) # scale to range 0 - 1
                rgb.data = (rgb.data * 255).astype(np.uint8) # convert to uint8
                io.save(os.path.join(pth, "rgb.hdr"), rgb )
                print(os.listdir(pth))
                self.assertTrue(os.path.exists(os.path.join(pth,'rgb.png')))

                # test loading png image from header file
                img = io.load(os.path.join(pth,'rgb.hdr'))
                self.assertTrue( img is not None)

                # test export to envi
                from BlackTelperion.io.libraries import saveLibraryTXT, loadLibraryTXT
                saveLibraryTXT( os.path.join(pth,"lib.txt"), self.lib )
                lib2 = loadLibraryTXT(  os.path.join(pth,"lib.txt") )
                self.assertTrue( (np.abs( self.lib.get_wavelengths() - lib2.get_wavelengths()) < 1e-5).all() )
                self.assertTrue((np.abs(self.lib.data - lib2.data) < 1e-5).all())

                # test save legend TODO
                # from BlackTelperion.analyse import saveLegend
                # saveLegend('Red stuff', 'Green stuff', 'Blue stuff', os.path.join(pth, 'legend.png'))
                # self.assertTrue(os.path.exists( os.path.join(pth, 'legend.png') ))

            except:
                shutil.rmtree(pth)  # delete temp directory
                self.assertFalse(True, "Error - could not save data of type %s" % str(type(data)))

            shutil.rmtree(pth)  # delete temp directory
    #TODO: Not implemented

    # def test_hycollection(self):
    #
    #     # load some data
    #     self.test_load()  # load datasets
    #     pth = mkdtemp()
    #     try:
    #         # build a HyCollection
    #         C = BlackTelperion.HyCollection( name = "testC", root = pth )
    #         C.img = self.img
    #         C.cld = self.cld
    #         C.lib = self.lib
    #         C.val = 100.
    #         C.arr = np.linspace(0,100)
    #         C.x = None # this should be ignored
    #         C.bool = True
    #
    #         # save it
    #         io.save( os.path.join(pth, "testC.hdr"), C )
    #         self.assertTrue(os.path.exists(os.path.join(os.path.join(pth, "testC.hyc"),"arr.npy")))  # check numpy array has been saved
    #         self.assertTrue(os.path.exists(os.path.join(os.path.join(pth, "testC.hyc"),"img.hdr")))  # check image has been saved
    #
    #         # load it
    #         C2 = io.load( os.path.join(pth, "testC.hdr") )
    #
    #         # test it
    #         self.assertEqual( C2.val, C.val )
    #         self.assertTrue( (C2.arr == C.arr).all() )
    #         self.assertTrue( C2.bool )
    #         self.assertEqual(C2.val, C.val)
    #         self.assertEqual(C2.val, C.val)
    #         self.assertEqual(C2.img.xdim(), C.img.xdim())
    #         self.assertEqual(C2.cld.point_count(), C.cld.point_count())
    #         self.assertEqual(C2.lib.sample_count(), C.lib.sample_count())
    #
    #         # test cleaning
    #         C2.bool = None
    #         C2.val = None
    #         C2.img = None
    #         C2.arr = None
    #         C2.clean()
    #
    #         self.assertFalse( 'bool' in C2.header )
    #         self.assertFalse('val' in C2.header)
    #         self.assertFalse(os.path.exists(os.path.join(os.path.join(pth, "testC.hyc"),"val.npy")))  # check numpy array has been deleted
    #         self.assertFalse(os.path.exists(os.path.join(os.path.join(pth, "testC.hyc"),"img.hdr"))) # check image has been deleted
    #
    #         # add a relative path!
    #         C2.addExternal( 'relobject', os.path.join(os.path.join(pth, "testC.hyc"),"lib.lib") )
    #         self.assertTrue( isinstance(C2.relobject, BlackTelperion.HyLibrary) )
    #
    #         # test saving collection in a different location
    #         io.save(os.path.join(pth, "testD.hdr"), C2 )
    #         self.assertTrue(os.path.exists(os.path.join(os.path.join(pth, "testD.hyc"),"cld.hdr")))  # check cloud has been copied across
    #
    #         # load this collection
    #         C3 = io.load(os.path.join(pth, "testD.hyc"))
    #         C3.inner = io.load( os.path.join(pth, "testC.hdr") ) # add nested collection
    #         C3.inner.arr2 = np.full( 40, 3.0 ) # add new thing to nested collection
    #         io.save(os.path.join(pth, "testE.hyc"), C3 )
    #         self.assertTrue(os.path.exists(os.path.join(os.path.join(pth, "testE.hyc"),"cld.hdr"))) # check cloud has been copied across
    #         self.assertTrue(os.path.exists(os.path.join(os.path.join(os.path.join(pth,
    #                                                     "testE.hyc"),"inner.hyc"),"cld.hdr"))) # check cloud has been copied across
    #         self.assertTrue(os.path.exists(os.path.join(os.path.join(os.path.join(pth,
    #                                                     "testE.hyc"),"inner.hyc"),"arr2.npy"))) # check cloud has been copied across
    #         self.assertTrue(isinstance(C3.relobject, BlackTelperion.HyLibrary)) # check relative path link can be loaded
    #
    #         # test quicksave function
    #         C3.save()
    #     except:
    #         shutil.rmtree(pth)  # delete temp directory
    #         self.assertFalse(True, "Error - could not create, load or save HyCollection." )
    #     shutil.rmtree(pth)  # delete temp directory
    #
    # def test_subset(self):
    #     from BlackTelperion.io.images import loadSubset
    #
    #     # load whole image for reference
    #     path = os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr")
    #     image = io.load(path)
    #
    #     # load subset and check that dimensions and values match
    #     subset = loadSubset(path, bands=BlackTelperion.SWIR )
    #     self.assertEqual(subset.xdim(), image.xdim())
    #     self.assertEqual(subset.ydim(), image.ydim())
    #     self.assertAlmostEqual(np.nanmax( np.abs(image.export_bands(BlackTelperion.SWIR).data - subset.data ) ), 0 )
    #
    #     # load a pixel and check that the dimensions and values match

if __name__ == '__main__':
    unittest.main()