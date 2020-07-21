# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
# from PIL import Image
# import numpy
import copy
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter, ImageChops
from itertools import permutations


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):  # input is a RvensProblem
        if problem.hasVisual:
            if problem.problemType == "3x3":
                A_img = Image.open(
                    problem.figures['A'].visualFilename,).convert('1')
                A = A_img.copy()
                # print(A.mode)
                B_img = Image.open(
                    problem.figures['B'].visualFilename, ).convert('1')
                B = B_img.copy()
                # print(B.mode)
                C_img = Image.open(
                    problem.figures['C'].visualFilename).convert('1')
                C = C_img.copy()
                D_img = Image.open(
                    problem.figures['D'].visualFilename).convert('1')
                D = D_img.copy()
                E_img = Image.open(
                    problem.figures['E'].visualFilename).convert('1')
                E = E_img.copy()
                F_img = Image.open(
                    problem.figures['F'].visualFilename).convert('1')
                F = F_img.copy()
                G_img = Image.open(
                    problem.figures['G'].visualFilename).convert('1')
                G = G_img.copy()
                H_img = Image.open(
                    problem.figures['H'].visualFilename).convert('1')
                H = H_img.copy()

                I = self.get_transformed_figure(A, B, C, D, E, F, G, H)
                # A_obj.show()
                # new_A_obj = A_obj.rotate(45)
                # new_A_obj_ref = ImageOps.mirror(A_obj)

                # best_similarity_measure(I,1,2,3,4,5,6,7,8)
                # I.show()
                best_sim = float('-inf')
                max_i = 0
                for i in range(8):
                    # print((problem.figures[str(i+1)].visualFilename))
                    prospective_I_img = Image.open(
                        problem.figures[str(i+1)].visualFilename).convert('L')
                    #print('i', i)
                    # prospective_I_img.show()
                    sim = self.get_similarity(I, prospective_I_img)
                    #print(i+1, sim)
                    if sim > best_sim:
                        best_sim = sim
                        max_i = i
                # print(max_i+1)
                return max_i+1
            else:
                return 1
        else:
            return 1

    def get_similarity(self, X, Y):
        npX = np.array(X)
        npY = np.array(Y)
        XinterY = np.maximum(npX, npY)
        XunionY = np.minimum(npX, npY)
        XdiffY = npX-npY
        YdiffX = npY-npX
        # print('sum', np.sum(XinterY), np.sum(XunionY))
        return np.sum(XinterY)/(np.sum(XinterY) + np.sum(XdiffY) + np.sum(YdiffX))

    def get_best_transform(self, X, Y):
        transform_dict = {1: "Identity", 2: '90', 3: '180',
                          4: '270', 5: 'flip', 6: '90f', 7: '180f', 8: '270f'}

        transform_simi_value = []
        # Identity
        transform_simi_value.append(self.get_similarity(X, Y))
        # Rotate 90
        img = X.rotate(90)
        transform_simi_value.append(self.get_similarity(img, Y))
        # Rotate 90
        img = X.rotate(180)
        transform_simi_value.append(self.get_similarity(img, Y))
        # Rotate 90
        img = X.rotate(270)
        transform_simi_value.append(self.get_similarity(img, Y))
        # flip
        flip_img = ImageOps.flip(X)
        transform_simi_value.append(self.get_similarity(flip_img, Y))
        # flip Rotate 90
        img = flip_img.rotate(90)
        transform_simi_value.append(self.get_similarity(img, Y))
        # flip Rotate 90
        img = flip_img.rotate(180)
        transform_simi_value.append(self.get_similarity(img, Y))
        # flip Rotate 90
        img = flip_img.rotate(270)
        transform_simi_value.append(self.get_similarity(img, Y))
        return transform_simi_value.index(max(transform_simi_value))+1, max(transform_simi_value)

    def apply_tranform(self, X, key):
        if key == 1:
            return X
        elif key == 2:
            return X.rotate(90)
        elif key == 3:
            return X.rotate(180)
        elif key == 4:
            return X.rotate(270)
        elif key == 5:
            new_X = ImageOps.flip(X)
            return new_X
        elif key == 6:
            new_X = ImageOps.flip(X)
            return new_X.rotate(90)
        elif key == 7:
            new_X = ImageOps.flip(X)
            return new_X.rotate(180)
        elif key == 8:
            new_X = ImageOps.flip(X)
            return new_X.rotate(270)

    def get_transformed_figure(self, A_old, B_old, C_old, D_old, E_old, F_old, G_old, H_old):
        # collinear tranform
        # apply transform to parallel
        # get the best fit image transformation among all the transformations.
        # get_fitness_value
        # ABC
        # DEF
        # GH?
        AB = ImageChops.logical_and(A_old, B_old).convert('L')
        DE = ImageChops.logical_and(D_old, E_old).convert('L')
        GH = ImageChops.logical_and(G_old, H_old).convert('L')
        AD = ImageChops.logical_and(A_old, D_old).convert('L')
        BE = ImageChops.logical_and(B_old, E_old).convert('L')
        CF = ImageChops.logical_and(C_old, F_old).convert('L')
        A = A_old.convert('L')
        B = B_old.convert('L')
        C = C_old.convert('L')
        D = D_old.convert('L')
        E = E_old.convert('L')
        F = F_old.convert('L')
        G = G_old.convert('L')
        H = H_old.convert('L')
        # AB = AB.convert('L')
        # DE = DE.convert('L')

        transform_dict = {1: 0, 2: 90, 3: 180,
                          4: 270, 5: 'flip', 6: '90f', 7: '180f', 8: '270f'}
        transform_key, similarity_value = self.get_best_transform(AB, C)
        best_value = similarity_value
        best_transform_key = transform_key
       # print('and', similarity_value)
        diff = ImageChops.difference(AB, C)
        horiz = 1
        diag = 0
        transform_key, similarity_value = self.get_best_transform(DE, F)
       # print('and', similarity_value)
        if similarity_value > best_value:
            best_value = similarity_value
            best_transform_key = transform_key
            diff = ImageChops.difference(DE, F)
        transform_key, similarity_value = self.get_best_transform(AD, G)
        #print('and', similarity_value)
        if similarity_value > best_value:
            best_value = similarity_value
            best_transform_key = transform_key
            horiz = 0
            diff = ImageChops.difference(AD, G)
        transform_key, similarity_value = self.get_best_transform(BE, H)
        #print('and', similarity_value)
        if similarity_value > best_value:
            best_value = similarity_value
            best_transform_key = transform_key
            horiz = 0
            diff = ImageChops.difference(BE, H)
        transform_key, similarity_value = self.get_best_transform(A, E)
       # print('and', similarity_value)
        if similarity_value > best_value:
            best_value = similarity_value
            best_transform_key = transform_key
            diag = 1
            diff = ImageChops.difference(A, E)

     #########################xor###############################
        xor = 0
        AB_xor = ImageChops.logical_xor(A_old, B_old)
        AB_xor = ImageChops.invert(AB_xor).convert('L')
        DE_xor = ImageChops.logical_xor(D_old, E_old)
        DE_xor = ImageChops.invert(DE_xor).convert('L')
        GH_xor = ImageChops.logical_xor(G_old, H_old)
        GH_xor = ImageChops.invert(GH_xor).convert('L')
        AD_xor = ImageChops.logical_xor(A_old, D_old)
        AD_xor = ImageChops.invert(AD_xor).convert('L')
        BE_xor = ImageChops.logical_xor(B_old, E_old)
        BE_xor = ImageChops.invert(BE_xor).convert('L')
        CF_xor = ImageChops.logical_xor(C_old, F_old)
        CF_xor = ImageChops.invert(CF_xor).convert('L')

        transform_dict = {1: 0, 2: 90, 3: 180,
                          4: 270, 5: 'flip', 6: '90f', 7: '180f', 8: '270f'}

        transform_key, similarity_value = self.get_best_transform(AB_xor, C)
        #print('pixel', np.array(AB_xor)[0][0])
        # AB_xor.show()
        # C.show()
        #print('xor', similarity_value)
        if similarity_value > best_value:
            best_value = similarity_value
            best_transform_key = transform_key

            diff = ImageChops.difference(AB_xor, C)
            horiz = 1
            xor = 1

        transform_key, similarity_value = self.get_best_transform(DE_xor, F)
        #print('xor', similarity_value)
        if similarity_value > best_value:
            best_value = similarity_value
            best_transform_key = transform_key
            diff = ImageChops.difference(DE_xor, F)
            horiz = 1
            xor = 1
        transform_key, similarity_value = self.get_best_transform(AD_xor, G)
        #print('xor', similarity_value)
        if similarity_value > best_value:
            best_value = similarity_value
            best_transform_key = transform_key
            horiz = 0
            xor = 1
            diff = ImageChops.difference(AD_xor, G)
        transform_key, similarity_value = self.get_best_transform(BE_xor, H)
        #print('xor', similarity_value)
        if similarity_value > best_value:
            best_value = similarity_value
            best_transform_key = transform_key
            horiz = 0
            xor = 1
            diff = ImageChops.difference(BE_xor, H)

        ##
        # horiz = 1
        # best_transform_key = transform_key
        #print('key', best_transform_key)
        # print(horiz)
        # print(diag)
        # print(xor)
        if diag == 1:
            old = E
        elif horiz == 1:
            if xor == 1:
                old = GH_xor
            else:
                old = GH
        else:
            if xor == 1:
                old = CF_xor
            else:
                old = CF
        # BE.show()
        I = self.apply_tranform(old, best_transform_key)
        # old.show()
        I = ImageChops.difference(I, diff)
        if diag == 1:
            diff2 = ImageChops.difference(H, D)
            I = ImageChops.difference(I, diff2)
        # I.show()
        return I
