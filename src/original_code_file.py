from __future__ import division
from __future__ import absolute_import
import time
from math import atan2, degrees
import math
from psychopy import data
from typing import BinaryIO
import csv
import os

# import Landolt_C as Stimulus_code
# import Landolt_C_HOAs as Stimulus_code_HOAs
import scipy
from scipy import ndimage, signal, interpolate
import readchar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import pygame

from pyfftw.interfaces.numpy_fft import rfft2, irfft2
from datetime import datetime
import matplotlib

matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)

# from core.stimulus import Stimulus
# from core.file_parser import FilesGetter
import zipfile
from lxml import etree
import pandas as pd
import glob
import os


class FilesGetter:
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        self.temp_result = list()
        self.file_reader = None

    def set_file_reader(self):
        if self.extension == "csv":
            self.file_reader = pd.read_csv
        elif self.extension == "xlsx":
            self.file_reader = pd.read_excel
        else:
            raise Exception(
                "We can't determin the file reader for such extension " + self.extension
            )

    @staticmethod
    def get_files(path, ext):
        if "." in ext:
            ext = ext.replace(".", "")
        return glob.glob(os.path.join(path, "*." + ext))

    def get_dir(self, path):
        folders = list()
        listOfFile = os.listdir(path)
        for entry in listOfFile:
            fullPath = self.get_full_path(path, entry)
            if os.path.isdir(fullPath):
                folders.append(fullPath)
        return folders

    @staticmethod
    def get_full_path(path, entry):
        return os.path.join(path, entry)

    def get_file_tree(self, path, ext):
        self.temp_result += self.get_files(path, ext)
        folders = self.get_dir(path)
        for folder in folders:
            return self.get_file_tree(folder, ext)

        temp = self.temp_result
        # self.temp_result = list()
        return temp

    def parse(self):
        return self.get_file_tree(self.path, self.extension)

    def read_data_from_files(self, files):
        data = {}
        for file in files:
            data[file] = self.file_reader(file, header=None).values.tolist()
        return data


stimulus = "Landolt_C"
observer = "KA"


class Stimulus:
    def __init__(
        self,
        condition,
        pupil_diameter,
        array_size,
        d_per_logMAR,
        test_stimulus="Landolt_C",
        dioptres_def=0,
        dioptres_ob_ast=0,
        dioptres_ver_ast=0,
        dioptres_ver_tre=0,
        dioptres_ver_com=0,
        dioptres_ob_tre=0,
        dioptres_ob_com=0,
        dioptres_sph=0,
        stimulus_intensity=1,
        background_intensity=0,
        colour_channel=1,
    ):
        self.condition = condition
        self.pupil_diameter = pupil_diameter
        print("pd = ", self.pupil_diameter)
        self.final_array_size = array_size
        self.m_final_array = np.min(self.final_array_size) / 2.0
        self.array_size = np.array([np.min(array_size), np.min(array_size)])
        self.final_d_per_logMAR = d_per_logMAR
        self.test_stimulus = test_stimulus

        self.dioptres_def = dioptres_def
        self.dioptres_ob_ast = dioptres_ob_ast
        self.dioptres_ver_ast = dioptres_ver_ast

        self.dioptres_ver_tre = dioptres_ver_tre
        self.dioptres_ver_com = dioptres_ver_com
        self.dioptres_ob_tre = dioptres_ob_tre
        self.dioptres_ob_com = dioptres_ob_com
        self.dioptres_sph = dioptres_sph

        self.stimulus_intensity = stimulus_intensity
        self.background_intensity = background_intensity
        self.colour_channel = colour_channel
        self.mx = (
            self.array_size[1]
        ) / 2.0  # np.median(np.arange(diameter)) # midpoint of array
        self.my = (self.array_size[0]) / 2.0
        self.X, self.Y = np.meshgrid(
            np.arange(-self.mx, self.mx), np.arange(-self.my, self.my)
        )  # This could be pre-calculated and held globally
        self.radial = self.X ** 2 + self.Y ** 2  # Could also be pre-calculated
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2) / self.mx
        self.Theta = np.angle(self.X + self.Y * 1j)
        self.final_coordinates = self.final_d_per_logMAR * np.arange(
            -self.m_final_array, self.m_final_array
        )

        self.padded_array_size = 768
        self.convolution_array_size = 768
        self.convolution_coordinates = self.final_d_per_logMAR * np.arange(
            -self.convolution_array_size / 2, self.convolution_array_size / 2
        )
        self.final_coordinates_padded = self.final_d_per_logMAR * np.arange(
            -self.padded_array_size / 2, self.padded_array_size / 2
        )
        self.wavelengths_nm = np.array([630, 520, 450])
        self.wavelengths_m = self.wavelengths_nm * 10 ** -9
        self.pupil = self.circle(self.array_size[1])

        self.defocus_array = self.calculate_zernike(2, 0)
        self.ob_ast_array = self.calculate_zernike(2, -2)
        self.ver_ast_array = self.calculate_zernike(2, 2)

        self.ver_tre_array = self.calculate_zernike(3, -3)
        self.ver_com_array = self.calculate_zernike(3, -1)
        self.ob_tre_array = self.calculate_zernike(3, 3)
        self.ob_com_array = self.calculate_zernike(3, 1)
        self.sph_array = self.calculate_zernike(4, 0)

        self.alpha = self.padded_array_size / np.min(array_size)
        print("alpha: ", self.alpha)

    def circle(self, diameter):
        """generates 2D array containing a circular aperture function with a diameter of diameter logMAR"""
        r = diameter / 2  # radius of circle
        circle = 1 * (
            self.radial < r ** 2
        )  # 1 for everything inside the circle, 0 for everything outside
        return circle

    def make_ring(self, diameter):
        circle_1 = self.circle(diameter)
        circle_2 = self.circle(3 * diameter / 5)
        circle_2 = np.abs(circle_2 - 1)
        ring = circle_1 * circle_2
        return ring

    def make_Landolt_C(self, orientation, diameter):
        ring = self.make_ring(diameter)
        gap_width = diameter / 5  # change the sisze of the gap of Landolt C
        print("orientation", orientation)
        if orientation == 0:
            gap = (
                1 * (self.Y > 0) * (self.X > -gap_width / 2) * (self.X < gap_width / 2)
            )
        if orientation == 1:
            gap = (
                1 * (self.X > 0) * (self.Y > -gap_width / 2) * (self.Y < gap_width / 2)
            )
        if orientation == 2:
            gap = (
                1 * (self.Y < 0) * (self.X > -gap_width / 2) * (self.X < gap_width / 2)
            )
        if orientation == 3:
            gap = (
                1 * (self.X < 0) * (self.Y > -gap_width / 2) * (self.Y < gap_width / 2)
            )
        gap = np.abs(gap - 1)
        C = gap * ring * (self.stimulus_intensity - self.background_intensity)
        return C

    def make_Single_point(self, diameter):
        # pixel_width = 10
        # pixel_width = 20
        logMAR_width = diameter
        stimulus = np.zeros((self.array_size))
        stimulus[
            int(round((self.my) - logMAR_width / 2)) : int(
                round((self.my) + logMAR_width / 2)
            ),
            int(round((self.mx) - logMAR_width / 2)) : int(
                round((self.mx) + logMAR_width / 2)
            ),
        ] = 1
        stimulus = stimulus * (self.stimulus_intensity - self.background_intensity)
        return stimulus

    def dioptres_to_RMS(self, dioptres):
        r = self.pupil_diameter / 2  # radius in m
        RMS = ((r ** 2) * dioptres) / (
            4 * math.sqrt(3)
        )  # calculation of RMS in m (RMS = root mean square error)
        RMS = RMS * (4 * math.sqrt(3)) / (r ** 2)
        return RMS

    def calculate_zernike(self, n, m):
        R = np.zeros((self.array_size))
        Z = np.zeros((self.array_size))
        abs_m = abs(m)
        for k in range(0, ((n - abs_m) // 2) + 1, 1):
            R += ((-1) ** k * math.factorial(n - k) * self.R ** (n - 2 * k)) / (
                math.factorial(k)
                * math.factorial((n + abs_m) / 2 - k)
                * math.factorial((n - abs_m) / 2 - k)
            )
        if m == 0:
            delta = 1.0
        else:
            delta = 0.0
        N = np.sqrt(2 * (n + 1) / (1 + delta))
        # print "N = ", N
        if m == 0:
            Z += N * R
        elif m < 0:
            Z += N * R * np.sin(abs_m * self.Theta)
        elif m > 0:
            Z += N * R * np.cos(abs_m * self.Theta)
        return Z

    def apply_defocus(self, dioptres_def):
        """generates 2D array with containing the phase distortions for defocus of dioptres_def dioptres"""
        defocus_array = self.defocus_array * self.dioptres_to_RMS(dioptres_def)
        return defocus_array

    def apply_ob_astigmatism(self, dioptres_ast):
        """generates 2D array with containing the phase distortions for defocus of dioptres_def dioptres"""
        astigmatism_array = self.ob_ast_array * self.dioptres_to_RMS(dioptres_ast)
        return astigmatism_array

    def apply_ver_astigmatism(self, dioptres_ast):
        """generates 2D array with containing the phase distortions for defocus of dioptres_def dioptres"""
        astigmatism_array = self.ver_ast_array * self.dioptres_to_RMS(dioptres_ast)
        return astigmatism_array

    #################################
    def apply_ver_tref(self, dioptres_ast):
        """generates 2D array with containing the phase distortions for defocus of dioptres_def dioptres"""
        astigmatism_array = self.ver_tre_array * self.dioptres_to_RMS(dioptres_ast)
        return astigmatism_array

    def apply_ver_coma(self, dioptres_ast):
        """generates 2D array with containing the phase distortions for defocus of dioptres_def dioptres"""
        astigmatism_array = self.ver_com_array * self.dioptres_to_RMS(dioptres_ast)
        return astigmatism_array

    def apply_ob_tref(self, dioptres_ast):
        """generates 2D array with containing the phase distortions for defocus of dioptres_def dioptres"""
        astigmatism_array = self.ob_tre_array * self.dioptres_to_RMS(dioptres_ast)
        return astigmatism_array

    def apply_ob_coma(self, dioptres_ast):
        """generates 2D array with containing the phase distortions for defocus of dioptres_def dioptres"""
        astigmatism_array = self.ob_com_array * self.dioptres_to_RMS(dioptres_ast)
        return astigmatism_array

    def apply_sphe(self, dioptres_ast):
        """generates 2D array with containing the phase distortions for defocus of dioptres_def dioptres"""
        astigmatism_array = self.sph_array * self.dioptres_to_RMS(dioptres_ast)
        return astigmatism_array

    ################################

    def mono_PSF(
        self,
        wavelength,
        dioptres_def,
        dioptres_ob_ast,
        dioptres_ver_ast,
        dioptres_ver_tre,
        dioptres_ver_com,
        dioptres_ob_tre,
        dioptres_ob_com,
        dioptres_sph,
    ):
        print(wavelength)
        r_per_logMAR = wavelength / (self.pupil_diameter * self.alpha)
        d_per_logMAR = r_per_logMAR * 180 / math.pi
        print("d_per_logMAR: ", d_per_logMAR)
        # if d_per_logMAR > self.final_d_per_logMAR:
        #   print ("Error: d_per_logMAR is too large (Landolt_C.py)")
        #  raise SystemExit()

        b_array_size = np.array(
            [
                int(np.rint(self.alpha * self.array_size[0])),
                int(np.rint(self.alpha * self.array_size[1])),
            ]
        )
        print("b_array_size: ", b_array_size)
        wavefront = (
            self.apply_defocus(dioptres_def)
            + self.apply_ob_astigmatism(dioptres_ob_ast)
            + self.apply_ver_astigmatism(dioptres_ver_ast)
            + self.apply_ver_tref(dioptres_ver_tre)
            + self.apply_ver_coma(dioptres_ver_com)
            + self.apply_ob_tref(dioptres_ob_tre)
            + self.apply_ob_coma(dioptres_ob_com)
            + self.apply_sphe(dioptres_sph)
        )

        theta = wavefront * (2 * math.pi) / (wavelength)
        # border the array of theta
        b_theta = np.zeros((b_array_size))
        b_theta[0 : self.array_size[0], 0 : self.array_size[1]] = theta
        dist = scipy.exp(1j * b_theta)

        # border the aperture array
        b_aperture = np.zeros((b_array_size))
        b_aperture[0 : self.array_size[0], 0 : self.array_size[1]] = self.pupil

        cw = b_aperture * dist

        print("ready for fft")
        fourier_transform = np.fft.fft2(cw)
        psf = np.abs(fourier_transform) ** 2  # take the modulus of the fft
        psf = np.fft.fftshift(psf)  # shift the 4 quadrants
        psf = psf / np.sum(psf)  # normalise the PSF to have a sum of 1

        print("PSF generated")

        print(d_per_logMAR)

        return psf, b_array_size, d_per_logMAR

    def make_fft_PSF(self):

        if self.condition == "rendered_mono":
            PSF, self.b_array_size, self.d_per_logMAR = self.mono_PSF(
                self.wavelengths_m[self.colour_channel],
                self.dioptres_def,
                self.dioptres_ob_ast,
                self.dioptres_ver_ast,
                self.dioptres_ver_tre,
                self.dioptres_ver_com,
                self.dioptres_ob_tre,
                self.dioptres_ob_com,
                self.dioptres_sph,
            )
            PSF = self.scale_PSF(PSF, self.b_array_size, self.d_per_logMAR)
            PSF = PSF / np.sum(PSF)

            fft_psf = rfft2(
                PSF,
                [int(self.convolution_array_size), int(self.convolution_array_size)],
            )
            # self.fft_psf = np.fft.fftshift(fft_psf)
            self.fft_psf = fft_psf

        elif self.condition == "rendered_chromablur":
            focus = 1.7312 - (
                (633.46) / (self.wavelengths_nm[1] - 214.10)
            )  # equation and peak focus from marimont and wandell
            red_LCA = (1.7312 - ((633.46) / (self.wavelengths_nm[0] - 214.10))) - focus
            green_LCA = 1.7312 - ((633.46) / (self.wavelengths_nm[1] - 214.10)) - focus
            blue_LCA = (1.7312 - ((633.46) / (self.wavelengths_nm[2] - 214.10))) - focus

            red_PSF, self.red_b_array_size, self.red_d_per_logMAR = self.mono_PSF(
                self.wavelengths_m[0],
                self.combine_def_LCA(self.dioptres_def, red_LCA),
                self.dioptres_ob_ast,
                self.dioptres_ver_ast,
                self.dioptres_ver_tre,
                self.dioptres_ver_com,
                self.dioptres_ob_tre,
                self.dioptres_ob_com,
                self.dioptres_sph,
            )
            red_PSF = self.scale_PSF(
                red_PSF, self.red_b_array_size, self.red_d_per_logMAR
            )
            red_PSF = red_PSF / np.sum(red_PSF)
            red_fft_psf = rfft2(red_PSF)
            # self.red_fft_psf = np.fft.fftshift(red_fft_psf)
            self.red_fft_psf = red_fft_psf

            green_PSF, self.green_b_array_size, self.green_d_per_logMAR = self.mono_PSF(
                self.wavelengths_m[1],
                self.combine_def_LCA(self.dioptres_def, green_LCA),
                self.dioptres_ob_ast,
                self.dioptres_ver_ast,
                self.dioptres_ver_tre,
                self.dioptres_ver_com,
                self.dioptres_ob_tre,
                self.dioptres_ob_com,
                self.dioptres_sph,
            )
            green_PSF = self.scale_PSF(
                green_PSF, self.green_b_array_size, self.green_d_per_logMAR
            )
            green_PSF = green_PSF / np.sum(green_PSF)
            green_fft_psf = rfft2(green_PSF)
            # self.green_fft_psf = np.fft.fftshift(green_fft_psf)
            self.green_fft_psf = green_fft_psf

            blue_PSF, self.blue_b_array_size, self.blue_d_per_logMAR = self.mono_PSF(
                self.wavelengths_m[2],
                self.combine_def_LCA(self.dioptres_def, blue_LCA),
                self.dioptres_ob_ast,
                self.dioptres_ver_ast,
                self.dioptres_ver_tre,
                self.dioptres_ver_com,
                self.dioptres_ob_tre,
                self.dioptres_ob_com,
                self.dioptres_sph,
            )
            blue_PSF = self.scale_PSF(
                blue_PSF, self.blue_b_array_size, self.blue_d_per_logMAR
            )
            blue_PSF = blue_PSF / np.sum(blue_PSF)
            blue_fft_psf = rfft2(blue_PSF)
            # self.blue_fft_psf = np.fft.fftshift(blue_fft_psf)
            self.blue_fft_psf = blue_fft_psf

    def scale_PSF(self, PSF, b_array_size, d_per_logMAR):
        coordinates_y = d_per_logMAR * (
            np.arange(-(b_array_size[0] - 1) / 2, ((b_array_size[0] - 1) / 2 + 1), 1)
        )
        coordinates_x = d_per_logMAR * (
            np.arange(-(b_array_size[1] - 1) / 2, ((b_array_size[1] - 1) / 2 + 1), 1)
        )
        # print 'x shape = ', coordinates_x.shape
        # print 'y shape = ', coordinates_y.shape
        # print 'PSF shape = ', PSF.shape
        f = interpolate.interp2d(coordinates_x, coordinates_y, PSF)
        # scaled_PSF = f(self.final_coordinates_padded, self.final_coordinates_padded)
        scaled_PSF = f(self.convolution_coordinates, self.convolution_coordinates)

        return scaled_PSF

    def blur_image(self, wavelength, Landolt_C, fft_psf):
        shape = Landolt_C.shape
        stimulus = (
            np.zeros((self.convolution_array_size, self.convolution_array_size))
            + self.background_intensity
        )
        stimulus[0 : shape[0], 0 : shape[1]] = Landolt_C
        print("total light stimulus:", np.sum(stimulus))
        b_image = irfft2(rfft2(stimulus) * fft_psf)
        b_image = np.fft.fftshift(b_image)
        image = b_image[: shape[0], : shape[1]]
        print("total light image:", np.sum(image))
        return image

    def combine_def_LCA(self, defocus, LCA):
        Ans = (defocus + LCA) ** 2 - LCA ** 2
        if Ans < 0:
            dioptres = 0
        else:
            dioptres = math.sqrt(Ans)
            if defocus < 0:
                dioptres = -dioptres
        return dioptres

    def make_stimulus(self, orientation, diameter, Landolt_C="empty"):
        if Landolt_C == "empty":
            if self.test_stimulus == "Landolt_C":
                Landolt_C = self.make_Landolt_C(orientation, diameter)
                Landolt_C = Landolt_C + self.background_intensity
            elif self.test_stimulus == "Single_point":
                Landolt_C = self.make_Single_point(diameter)
                Landolt_C = Landolt_C + self.background_intensity

        self.stimulus = np.zeros(
            (self.final_array_size[0], self.final_array_size[1], 3)
        )
        self.stimulus[:, :, self.colour_channel] = self.background_intensity
        padding = int((np.max(self.final_array_size) - np.max(Landolt_C.shape)) / 2)
        print("padding = ", padding)
        print("C shape", Landolt_C.shape)
        print("stimulus shape", self.stimulus.shape)

        if self.condition == "optical_mono":
            # self.stimulus[:,:,self.colour_channel] = self.background_intensity

            if padding == 0:
                self.stimulus[:, :, self.colour_channel] = Landolt_C
            else:
                # self.stimulus[padding:-padding,:,self.colour_channel] = Landolt_C
                self.stimulus[padding:-padding, :, self.colour_channel] = Landolt_C

            self.stimulus[:, :, self.colour_channel] = self.stimulus[
                :, :, self.colour_channel
            ]

            print("total light image:", np.sum(self.stimulus))
        elif self.condition == "optical_white":
            # self.stimulus[:,:,:] = self.background_intensity
            if padding == 0:
                self.stimulus[:, :, 0] = Landolt_C
                self.stimulus[:, :, 1] = Landolt_C
                self.stimulus[:, :, 2] = Landolt_C
            else:
                self.stimulus[padding:-padding, :, 0] = Landolt_C
                self.stimulus[padding:-padding, :, 1] = Landolt_C
                self.stimulus[padding:-padding, :, 2] = Landolt_C
            self.stimulus[:, :, :] = self.stimulus[:, :, :]
        elif self.condition == "rendered_mono":
            # self.stimulus[:,:,self.colour_channel] = self.background_intensity

            if padding == 0:
                self.stimulus[:, :, self.colour_channel] = self.blur_image(
                    self.wavelengths_m[self.colour_channel], Landolt_C, self.fft_psf
                )
            else:
                self.stimulus[
                    padding:-padding, :, self.colour_channel
                ] = self.blur_image(
                    self.wavelengths_m[self.colour_channel], Landolt_C, self.fft_psf
                )
            self.stimulus[:, :, self.colour_channel] = self.stimulus[
                :, :, self.colour_channel
            ]
        elif self.condition == "rendered_chromablur":
            # self.stimulus[:,:,:] = self.background_intensity
            if padding == 0:
                self.stimulus[:, :, 0] = self.blur_image(
                    self.wavelengths_m[0], Landolt_C, self.red_fft_psf
                )
                self.stimulus[:, :, 1] = self.blur_image(
                    self.wavelengths_m[1], Landolt_C, self.green_fft_psf
                )
                self.stimulus[:, :, 2] = self.blur_image(
                    self.wavelengths_m[2], Landolt_C, self.blue_fft_psf
                )
            else:
                self.stimulus[padding:-padding, :, 0] = self.blur_image(
                    self.wavelengths_m[0], Landolt_C, self.red_fft_psf
                )
                self.stimulus[padding:-padding, :, 1] = self.blur_image(
                    self.wavelengths_m[1], Landolt_C, self.green_fft_psf
                )
                self.stimulus[padding:-padding, :, 2] = self.blur_image(
                    self.wavelengths_m[2], Landolt_C, self.blue_fft_psf
                )
            self.stimulus[:, :, :] = self.stimulus[:, :, :]

        print("max_value_stimulus", np.max(self.stimulus))

        return self.stimulus


def Degree2Pix(M_height, distance, v_res, size_deg):  # Convert Degree to Pixel
    d_pp = degrees(atan2(M_height, distance)) / (v_res)
    piels = size_deg / d_pp
    return piels


def arcMin2Dedegree(am):  # convert arcmine to degree
    return am * 0.01667


def degree2arcMin(deg):  # convert degree to arcmine
    return deg * 60


def arcMin2LogMar(am):  # convert arcmine to logMAR
    return math.log10(am / 5)


correct_ans = 0

import pandas as pd


def get_result_for_one_iter(arc, user_res):
    """calcluate the result for one iteration"""
    deg = arcMin2Dedegree(arc)
    dia = Degree2Pix(26.5, 375, 768, deg)
    LogM = arcMin2LogMar(arc)

    # five values [LogM, deg, arc, dia, user_res]
    return {"LogM": LogM, "deg": deg, " arc": arc, "dia": dia, "user_res": user_res}


def Write_Result(arc, user_res):
    deg = arcMin2Dedegree(arc)
    dia = Degree2Pix(26.5, 375, 768, deg)
    LogM = arcMin2LogMar(arc)

    print("write")
    global result_folder
    f = open(result_folder + FileName, "a")  # type:
    f.write(
        str(LogM)
        + "\t"
        + "\t"
        + "\t"
        + str(deg)
        + "\t"
        + "\t"
        + "\t"
        + str(arc)
        + "\t"
        + "\t"
        + "\t"
        + str(dia)
        + "\t"
        + "\t"
        + "\t"
        + str(user_res)
        + "\n"
    )
    f.close()


def QuestStairCase():
    global FinalResult
    FinalResult = list()
    guessGapArcmin = 5.5
    guessSD = 5
    minArcMin = 0.3
    maxArcMin = 8
    staircase = data.QuestHandler(
        guessGapArcmin,
        guessSD,
        pThreshold=0.63,
        gamma=0.25,
        method="mean",
        nTrials=15,
        minVal=minArcMin,
        maxVal=maxArcMin,
    )
    for thisGapArcmin in staircase:
        # staircase.range
        print(thisGapArcmin)
        # 1) get the diameter of c in degrees
        Arcmin = thisGapArcmin * 5
        # 2)Now Arcmin to degree
        degree = arcMin2Dedegree(Arcmin)
        # 3) Now degrees to pixel mapping
        dia = Degree2Pix(26.5, 375, 768, degree)
        # 4) Now rand orientation
        orient = random.randrange(0, 3)
        Landolt_C1 = C.make_stimulus(orient, dia, Landolt_C="empty")
        Landolt_C = np.zeros((1024, 768))
        Landolt_C[:, :] = Landolt_C1[:, :, 1]
        if flag != 1:
            chk1 = signal.convolve(Landolt_C, combined_psf, mode="same")
        if flag == 1:
            chk1 = Landolt_C
        chk = np.zeros((1024, 768, 3))
        chk[:, :, 1] = chk1
        chk = chk / chk.max()

        chk *= 255
        my_surface = pygame.surfarray.make_surface(chk)
        display_surface.blit(my_surface, (0, 0))
        pygame.display.flip()
        running = True
        while running:
            e = pygame.event.wait()
            if e.type == pygame.KEYDOWN:
                if (
                    e.key == pygame.K_DOWN
                ):  # if Landolt C orientation is up press arrow up
                    running = False
                    if orient == 3:  # up = 3
                        print("correct")
                        #
                        FinalResult.append(get_result_for_one_iter(Arcmin, 1))
                        Write_Result(
                            Arcmin, 1
                        )  # if the result is correct print number 1
                        staircase.addResponse(1)
                    else:
                        print("incorrect")
                        FinalResult.append(get_result_for_one_iter(Arcmin, 0))
                        Write_Result(
                            Arcmin, 0
                        )  # if the result is incorrect print number 0
                        staircase.addResponse(0)
                elif (
                    e.key == pygame.K_UP
                ):  # if Landolt C orientation is down press arrow down
                    running = False
                    if orient == 1:  # down = 1
                        print("correct")
                        FinalResult.append(get_result_for_one_iter(Arcmin, 1))
                        Write_Result(Arcmin, 1)
                        staircase.addResponse(1)
                    else:
                        print("incorrect")
                        FinalResult.append(get_result_for_one_iter(Arcmin, 0))
                        Write_Result(Arcmin, 0)
                        staircase.addResponse(0)
                elif (
                    e.key == pygame.K_LEFT
                ):  # if Landolt C orientation is right press arrow right
                    running = False
                    if orient == 0:  # right = 0
                        print("correct")
                        FinalResult.append(get_result_for_one_iter(Arcmin, 1))
                        Write_Result(Arcmin, 1)
                        staircase.addResponse(1)
                    else:
                        print("incorrect")
                        FinalResult.append(get_result_for_one_iter(Arcmin, 0))
                        Write_Result(Arcmin, 0)
                        staircase.addResponse(0)
                elif (
                    e.key == pygame.K_RIGHT
                ):  # if Landolt C orientation is left press arrow left
                    running = False
                    if orient == 2:  # left = 2
                        print("correct")
                        FinalResult.append(get_result_for_one_iter(Arcmin, 1))
                        Write_Result(Arcmin, 1)
                        staircase.addResponse(1)
                    else:
                        print("incorrect")
                        FinalResult.append(get_result_for_one_iter(Arcmin, 0))
                        Write_Result(Arcmin, 0)
                        staircase.addResponse(0)
                elif e.key == pygame.K_ESCAPE:
                    running = False
                    pygame.quit()
                else:
                    print("Invalid Input")


def experiment(filename, flag, result_folder):

    diameter = 166  # Size of Landlot C in pixel
    condition = "rendered_mono"  # change the condition ( optical or rendered_mono or rendered_chromablur)
    pupil_diameter_mm = 4
    pupil_diameter = pupil_diameter_mm / 1000
    array_size = np.array([1024, 768])  # screen size
    orientation = 1
    # d_per_Pixel = 0.01
    d_per_Pixel = degrees(atan2(26.5, 375)) / 768
    Size_in_degree = 2  # start point of Landlot C size in degree
    max_deg_size = 1.5  # max size of landoltC where code will exit
    max_pix_size = max_deg_size / d_per_Pixel
    max_fail_iter = 8  # set maximum number of mistakes at which pygame will exit
    fail_iter_count = 0
    global C
    C = Stimulus(
        condition,
        pupil_diameter,
        array_size,
        d_per_Pixel,
        stimulus_intensity=0.8,
        background_intensity=0.2,
    )
    C.make_fft_PSF()
    X1_time = time.time()
    #####################################

    # choose flag=1 if only object is required
    # choose flag=2 if object and stimulus blur is required
    # choose flag=3 if object and observer blur is required
    # choose flag=4 if object, stimulus blur and observer blur is required.
    # flag = 1   # set flag here

    pupil_diameter_stimulus_mm = 5  # change stimulus aperture here
    pupil_diameter_observer_mm = 5  # change observer aperture here

    defocus_stimulus = 1  # add your additional defocus here for stimulus
    defocus_observer = 1  # add your additional defocus here for observer

    stimulus_hoa = 1  # if stiulus hoa is required put 1 else put 0
    observer_hoa = 1  # if observer hoa is required put 1 else put 0

    pupil_diameter_stimulus = pupil_diameter_stimulus_mm / 1000
    pupil_diameter_observer = pupil_diameter_observer_mm / 1000
    global LC
    LC = Stimulus(
        condition,
        pupil_diameter_stimulus,
        array_size,
        d_per_Pixel,
        test_stimulus="Landolt_C",
        dioptres_def=0,
        dioptres_ob_ast=0,
        dioptres_ver_ast=0,
        dioptres_ver_tre=0,
        dioptres_ver_com=0,
        dioptres_ob_tre=0,
        dioptres_ob_com=0,
        dioptres_sph=0,
        stimulus_intensity=1,
        background_intensity=0,
    )
    Landolt_C = LC.make_Landolt_C(orientation, diameter)

    rows = pd.read_csv(filename, header=None).values.tolist()
    frame = rows[7][7:15]  # change frame number from here
    print(frame)

    wav_diop = 500 * 10 ** -9
    defocus_offset_stimulus = (
        ((pupil_diameter_stimulus / 2) ** 2) * defocus_stimulus
    ) / (4 * math.sqrt(3))
    defocus_offset_observer = (
        ((pupil_diameter_observer / 2) ** 2) * defocus_observer
    ) / (4 * math.sqrt(3))

    # This is HOA taken from excel file
    XL_astig_V = float(frame[0]) * wav_diop
    XL_defocus = float(frame[1]) * wav_diop
    XL_astig_O = float(frame[2]) * wav_diop
    XL_tref_V = float(frame[3]) * wav_diop
    XL_Coma_V = float(frame[4]) * wav_diop
    XL_tref_O = float(frame[5]) * wav_diop
    XL_coma_O = float(frame[6]) * wav_diop
    XL_spher = float(frame[7]) * wav_diop
    global SStimulus
    array_size = np.array([200, 200])
    SStimulus = Stimulus(
        condition,
        pupil_diameter_stimulus,
        array_size,
        d_per_Pixel,
        test_stimulus="Single_point",
        dioptres_def=XL_defocus * stimulus_hoa + defocus_offset_stimulus,
        dioptres_ob_ast=XL_astig_O * stimulus_hoa,
        dioptres_ver_ast=XL_astig_V * stimulus_hoa,
        dioptres_ver_tre=XL_tref_V * stimulus_hoa,
        dioptres_ver_com=XL_Coma_V * stimulus_hoa,
        dioptres_ob_tre=XL_tref_O * stimulus_hoa,
        dioptres_ob_com=XL_coma_O * stimulus_hoa,
        dioptres_sph=XL_spher * stimulus_hoa,
        stimulus_intensity=1,
        background_intensity=0,
    )
    SStimulus.make_fft_PSF()
    diameter = 2
    SStimulus_PSF = SStimulus.make_stimulus(orientation, diameter, Landolt_C="empty")
    SStimulus_PSF_max = SStimulus_PSF.max()
    SStimulus_PSF_single = np.zeros((200, 200))
    SStimulus_PSF_single[:, :] = SStimulus_PSF[:, :, 1]
    global OObserver
    OObserver = Stimulus(
        condition,
        pupil_diameter_observer,
        array_size,
        d_per_Pixel,
        test_stimulus="Single_point",
        dioptres_def=XL_defocus * observer_hoa + defocus_offset_observer,
        dioptres_ob_ast=XL_astig_O * observer_hoa,
        dioptres_ver_ast=XL_astig_V * observer_hoa,
        dioptres_ver_tre=XL_tref_V * observer_hoa,
        dioptres_ver_com=XL_Coma_V * observer_hoa,
        dioptres_ob_tre=XL_tref_O * observer_hoa,
        dioptres_ob_com=XL_coma_O * observer_hoa,
        dioptres_sph=XL_spher * observer_hoa,
        stimulus_intensity=1,
        background_intensity=0,
    )
    OObserver.make_fft_PSF()
    diameter = 2
    OObserver_PSF = OObserver.make_stimulus(orientation, diameter, Landolt_C="empty")
    OObserver_PSF_max = OObserver_PSF.max()
    OObserver_PSF_single = np.zeros((200, 200))
    OObserver_PSF_single[:, :] = OObserver_PSF[:, :, 1]

    global combined_psf

    if flag == 1:
        combined_psf = SStimulus_PSF_single
    if flag == 2:
        combined_psf = SStimulus_PSF_single
    if flag == 3:
        combined_psf = OObserver_PSF_single
    if flag == 4:
        combined_psf = signal.convolve2d(
            SStimulus_PSF_single, OObserver_PSF_single, boundary="symm", mode="same"
        )

    diameter = 32.8115713859
    array_size = np.array([1024, 768])
    global FileName
    global now
    now = datetime.now()
    FileName = (
        "KhattarExp_" + now.strftime("%H_%M_%S") + ".txt"
    )  # KhatarExp file to save time and data on every experiment.
    f = open(
        result_folder + FileName, "w"
    )  # type: # Results and Graphs folder has KhatarExp file.
    f.write(
        "logMAR"
        + "\t"
        + "\t"
        + "\t"
        + "\t"
        + "degree"
        + "\t"
        + "\t"
        + "\t"
        + "\t"
        + "\t"
        + "arcmin"
        + "\t"
        + "\t"
        + "\t"
        + "\t"
        + "\t"
        + "Pixels"
        + "\t"
        + "\t"
        + "\t"
        + "\t"
        + "Observation"
        + "\n"
    )
    f.close()  # data units

    pygame.init()

    X = 1024  # original was 1024
    Y = 768
    global display_surface
    display_surface = pygame.display.set_mode((X, Y))

    pygame.init()
    pygame.display.set_mode((1024, 768), pygame.FULLSCREEN)

    # pygame.display.set_mode((1024, 768),pygame.NOFRAME )

    pygame.display.flip()

    pygame.display.set_caption("Khatar Staircase Experiment")
    white = (255, 255, 255)
    green = (0, 255, 0)
    black = (0, 0, 0)
    font = pygame.font.Font("freesansbold.ttf", 32)
    text = font.render("IF YOU ARE READY, PRESS ENTER", True, green, black)
    text = pygame.transform.rotate(text, 180)
    textRect = text.get_rect()
    textRect.center = (X // 2, Y // 2)
    Trial = 0
    display_surface.blit(text, textRect)
    pygame.display.flip()
    running = True
    # print(pygame.display.Info())
    while running:
        e = pygame.event.wait()
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RETURN:
                running = False
                QuestStairCase()
            else:
                pygame.quit()
    global FinalResult
    MyFinalResult = pd.DataFrame(FinalResult)
    # save result ot csv  [it is east to retraive the data from a csv file ]
    csv_result_filename = "KhattarExp_" + now.strftime("%H_%M_%S") + ".csv"
    MyFinalResult.to_csv(result_folder + csv_result_filename, index=False)
    print(MyFinalResult.head())
    # response_List = []
    # with open(result_folder + FileName) as f:
    #     next(f)
    #     for line in f:
    #         val = line.split()[-1]
    #         response_List.append(val)
    # print(response_List)
    # LogMArList = []
    # # you are getten the list from the result file , let's take a look
    # with open(result_folder + FileName) as f:
    #     next(f)
    #     for line in f:
    #         Line = line.split()
    #         val = float(Line[0])
    #         #LogMARList.append(val)
    #         #LogMArList.append(str(round(val, 3)))
    #         LogMArList.append(val)
    #
    #         LogMArList.append(val)
    #         LogMArList.sort()
    # # for some reasons LogMArList is empty list let's check why
    # print(LogMArList)
    #
    # TrialList = []
    # for x in range(1, len(LogMArList) + 1):
    #     TrialList.append(x)
    #
    # # TrialList is just an integer number from 1 to the length of LogMArList
    # # LogMArList is the most important list
    # TrialList = TrialList[::-1]
    # plot_figure(TrialList, LogMArList, response_List, result_folder)

    # plot from the dataframe
    response_List = list(map(str, MyFinalResult["user_res"].to_list()))
    LogMArList = MyFinalResult["LogM"].to_list()
    TrialList = list(range(1, len(LogMArList) + 1))
    plot_figure(TrialList, LogMArList, response_List, result_folder)


def plot_figure(TrialList, LogMArList, response_List, result_folder):
    # plt.style.use('seaborn-whitegrid')
    fig1 = plt.gcf()
    plt.title("Staircase Experiment Results(Optical)")
    plt.xlabel("Trial Number")
    plt.ylabel("LogMAR")
    # x is TrialList
    # y is LogMArList
    # let's see how did you calculate them
    plt.plot(TrialList, LogMArList, color="black")
    mixed = False
    if len(set(response_List)) == 1:
        res = all(x == "1" for x in response_List)
        if res == True:
            plt.plot(TrialList, LogMArList, marker="s", color="black")
        else:
            plt.plot(
                TrialList,
                LogMArList,
                marker="s",
                color="black",
                markerfacecolor="none",
                markeredgecolor="k",
            )
    else:
        if "0" in response_List and "1" in response_List:
            xs = [a for a, b in zip(TrialList, response_List) if b == "0"]
            ys = [a for a, b in zip(LogMArList, response_List) if b == "0"]
            plt.scatter(xs, ys, marker="s", facecolor="none", edgecolors="k")

            xt = [a for a, b in zip(TrialList, response_List) if b == "1"]
            yt = [a for a, b in zip(LogMArList, response_List) if b == "1"]
            plt.scatter(xt, yt, marker="s", facecolor="k")
        else:
            plt.scatter(TrialList, LogMArList, marker=".", s=0)
    global now
    fig1 = plt.gcf()
    plt.show()
    FigName = "KhattarExp_" + now.strftime("%H_%M_%S") + ".png"
    fig1.savefig(result_folder + FigName)


global C, OObserver, SStimulus, display_surface
# save the final result in this var(a dataframe or a list of dict)
global FinalResult
if __name__ == "__main__":
    from os.path import join

    file = join(# "..", 
                "data", "Eye_03.csv")

    # file = "C:/Users/admin-nreadlab/Desktop/eyes/Khatar_right_logfile_20_03_12_13_29_20 (1).csv"
    flag = 2
    result_folder = join(# "..", 
                         "report", "figures") + '/'
    experiment(file, flag, result_folder)
