#!/usr/bin/env python
# Generated by gen_gaunt_data.py

import numpy as np

GAUNT_LMAX = 5

# (l1, l2, l3, m1, m2, m3) : G
GAUNT_COEFF_DATA1 = {
   ( 0, 0, 0,  0, 0, 0) :   0.2820947917738781,   ( 0, 1, 1,  0,-1, 1) :  -0.2820947917738781,   ( 0, 1, 1,  0, 0, 0) :   0.2820947917738781,
   ( 0, 1, 1,  0, 1,-1) :  -0.2820947917738781,   ( 0, 2, 2,  0,-2, 2) :   0.2820947917738781,   ( 0, 2, 2,  0,-1, 1) :  -0.2820947917738781,
   ( 0, 2, 2,  0, 0, 0) :   0.2820947917738781,   ( 0, 2, 2,  0, 1,-1) :  -0.2820947917738781,   ( 0, 2, 2,  0, 2,-2) :   0.2820947917738781,
   ( 0, 3, 3,  0,-3, 3) :  -0.2820947917738781,   ( 0, 3, 3,  0,-2, 2) :   0.2820947917738781,   ( 0, 3, 3,  0,-1, 1) :  -0.2820947917738781,
   ( 0, 3, 3,  0, 0, 0) :   0.2820947917738781,   ( 0, 3, 3,  0, 1,-1) :  -0.2820947917738781,   ( 0, 3, 3,  0, 2,-2) :   0.2820947917738781,
   ( 0, 3, 3,  0, 3,-3) :  -0.2820947917738781,   ( 0, 4, 4,  0,-4, 4) :   0.2820947917738781,   ( 0, 4, 4,  0,-3, 3) :  -0.2820947917738781,
   ( 0, 4, 4,  0,-2, 2) :   0.2820947917738781,   ( 0, 4, 4,  0,-1, 1) :  -0.2820947917738781,   ( 0, 4, 4,  0, 0, 0) :   0.2820947917738781,
   ( 0, 4, 4,  0, 1,-1) :  -0.2820947917738781,   ( 0, 4, 4,  0, 2,-2) :   0.2820947917738781,   ( 0, 4, 4,  0, 3,-3) :  -0.2820947917738781,
   ( 0, 4, 4,  0, 4,-4) :   0.2820947917738781,   ( 1, 1, 2, -1,-1, 2) :   0.3090193616185516,   ( 1, 1, 2, -1, 0, 1) :  -0.2185096861184158,
   ( 1, 1, 2, -1, 1, 0) :   0.1261566261010080,   ( 1, 1, 2,  0,-1, 1) :  -0.2185096861184158,   ( 1, 1, 2,  0, 0, 0) :   0.2523132522020160,
   ( 1, 1, 2,  0, 1,-1) :  -0.2185096861184158,   ( 1, 1, 2,  1,-1, 0) :   0.1261566261010080,   ( 1, 1, 2,  1, 0,-1) :  -0.2185096861184158,
   ( 1, 1, 2,  1, 1,-2) :   0.3090193616185516,   ( 1, 2, 3, -1,-2, 3) :  -0.3198654279343846,   ( 1, 2, 3, -1,-1, 2) :   0.2611690282654090,
   ( 1, 2, 3, -1, 0, 1) :  -0.2023006594034206,   ( 1, 2, 3, -1, 1, 0) :   0.1430481681026688,   ( 1, 2, 3, -1, 2,-1) :  -0.0825888983611587,
   ( 1, 2, 3,  0,-2, 2) :   0.1846743909223718,   ( 1, 2, 3,  0,-1, 1) :  -0.2335966803276074,   ( 1, 2, 3,  0, 0, 0) :   0.2477666950834761,
   ( 1, 2, 3,  0, 1,-1) :  -0.2335966803276074,   ( 1, 2, 3,  0, 2,-2) :   0.1846743909223718,   ( 1, 2, 3,  1,-2, 1) :  -0.0825888983611587,
   ( 1, 2, 3,  1,-1, 0) :   0.1430481681026688,   ( 1, 2, 3,  1, 0,-1) :  -0.2023006594034206,   ( 1, 2, 3,  1, 1,-2) :   0.2611690282654090,
   ( 1, 2, 3,  1, 2,-3) :  -0.3198654279343846,   ( 1, 3, 4, -1,-3, 4) :   0.3257350079352799,   ( 1, 3, 4, -1,-2, 3) :  -0.2820947917738781,
   ( 1, 3, 4, -1,-1, 2) :   0.2384136135044481,   ( 1, 3, 4, -1, 0, 1) :  -0.1946639002730062,   ( 1, 3, 4, -1, 1, 0) :   0.1507860087730269,
   ( 1, 3, 4, -1, 2,-1) :  -0.1066218093114615,   ( 1, 3, 4, -1, 3,-2) :   0.0615581303074573,   ( 1, 3, 4,  0,-3, 3) :  -0.1628675039676400,
   ( 1, 3, 4,  0,-2, 2) :   0.2132436186229231,   ( 1, 3, 4,  0,-1, 1) :  -0.2384136135044481,   ( 1, 3, 4,  0, 0, 0) :   0.2462325212298291,
   ( 1, 3, 4,  0, 1,-1) :  -0.2384136135044481,   ( 1, 3, 4,  0, 2,-2) :   0.2132436186229231,   ( 1, 3, 4,  0, 3,-3) :  -0.1628675039676400,
   ( 1, 3, 4,  1,-3, 2) :   0.0615581303074573,   ( 1, 3, 4,  1,-2, 1) :  -0.1066218093114615,   ( 1, 3, 4,  1,-1, 0) :   0.1507860087730269,
   ( 1, 3, 4,  1, 0,-1) :  -0.1946639002730062,   ( 1, 3, 4,  1, 1,-2) :   0.2384136135044481,   ( 1, 3, 4,  1, 2,-3) :  -0.2820947917738781,
   ( 1, 3, 4,  1, 3,-4) :   0.3257350079352799,   ( 2, 2, 2, -2, 0, 2) :  -0.1802237515728686,   ( 2, 2, 2, -2, 1, 1) :   0.2207281154418226,
   ( 2, 2, 2, -2, 2, 0) :  -0.1802237515728686,   ( 2, 2, 2, -1,-1, 2) :   0.2207281154418226,   ( 2, 2, 2, -1, 0, 1) :  -0.0901118757864343,
   ( 2, 2, 2, -1, 1, 0) :  -0.0901118757864343,   ( 2, 2, 2, -1, 2,-1) :   0.2207281154418226,   ( 2, 2, 2,  0,-2, 2) :  -0.1802237515728686,
   ( 2, 2, 2,  0,-1, 1) :  -0.0901118757864343,   ( 2, 2, 2,  0, 0, 0) :   0.1802237515728686,   ( 2, 2, 2,  0, 1,-1) :  -0.0901118757864343,
   ( 2, 2, 2,  0, 2,-2) :  -0.1802237515728686,   ( 2, 2, 2,  1,-2, 1) :   0.2207281154418226,   ( 2, 2, 2,  1,-1, 0) :  -0.0901118757864343,
   ( 2, 2, 2,  1, 0,-1) :  -0.0901118757864343,   ( 2, 2, 2,  1, 1,-2) :   0.2207281154418226,   ( 2, 2, 2,  2,-2, 0) :  -0.1802237515728686,
   ( 2, 2, 2,  2,-1,-1) :   0.2207281154418226,   ( 2, 2, 2,  2, 0,-2) :  -0.1802237515728686,   ( 2, 2, 4, -2,-2, 4) :   0.3371677656723677,
   ( 2, 2, 4, -2,-1, 3) :  -0.2384136135044481,   ( 2, 2, 4, -2, 0, 2) :   0.1560783472274399,   ( 2, 2, 4, -2, 1, 1) :  -0.0901118757864343,
   ( 2, 2, 4, -2, 2, 0) :   0.0402992559676969,   ( 2, 2, 4, -1,-2, 3) :  -0.2384136135044481,   ( 2, 2, 4, -1,-1, 2) :   0.2548748737361102,
   ( 2, 2, 4, -1, 0, 1) :  -0.2207281154418226,   ( 2, 2, 4, -1, 1, 0) :   0.1611970238707875,   ( 2, 2, 4, -1, 2,-1) :  -0.0901118757864343,
   ( 2, 2, 4,  0,-2, 2) :   0.1560783472274399,   ( 2, 2, 4,  0,-1, 1) :  -0.2207281154418226,   ( 2, 2, 4,  0, 0, 0) :   0.2417955358061813,
   ( 2, 2, 4,  0, 1,-1) :  -0.2207281154418226,   ( 2, 2, 4,  0, 2,-2) :   0.1560783472274399,   ( 2, 2, 4,  1,-2, 1) :  -0.0901118757864343,
   ( 2, 2, 4,  1,-1, 0) :   0.1611970238707875,   ( 2, 2, 4,  1, 0,-1) :  -0.2207281154418226,   ( 2, 2, 4,  1, 1,-2) :   0.2548748737361102,
   ( 2, 2, 4,  1, 2,-3) :  -0.2384136135044481,   ( 2, 2, 4,  2,-2, 0) :   0.0402992559676969,   ( 2, 2, 4,  2,-1,-1) :  -0.0901118757864343,
   ( 2, 2, 4,  2, 0,-2) :   0.1560783472274399,   ( 2, 2, 4,  2, 1,-3) :  -0.2384136135044481,   ( 2, 2, 4,  2, 2,-4) :   0.3371677656723677,
   ( 2, 3, 3, -2,-1, 3) :   0.1329807601338109,   ( 2, 3, 3, -2, 0, 2) :  -0.1880631945159188,   ( 2, 3, 3, -2, 1, 1) :   0.2060129077457011,
   ( 2, 3, 3, -2, 2, 0) :  -0.1880631945159188,   ( 2, 3, 3, -2, 3,-1) :   0.1329807601338109,   ( 2, 3, 3, -1,-2, 3) :  -0.2102610435016800,
   ( 2, 3, 3, -1,-1, 2) :   0.1628675039676400,   ( 2, 3, 3, -1, 0, 1) :  -0.0594708038717590,   ( 2, 3, 3, -1, 1, 0) :  -0.0594708038717590,
   ( 2, 3, 3, -1, 2,-1) :   0.1628675039676400,   ( 2, 3, 3, -1, 3,-2) :  -0.2102610435016800,   ( 2, 3, 3,  0,-3, 3) :   0.2102610435016800,
   ( 2, 3, 3,  0,-2, 2) :   0.0000000000000000,   ( 2, 3, 3,  0,-1, 1) :  -0.1261566261010080,   ( 2, 3, 3,  0, 0, 0) :   0.1682088348013440,
   ( 2, 3, 3,  0, 1,-1) :  -0.1261566261010080,   ( 2, 3, 3,  0, 2,-2) :   0.0000000000000000,   ( 2, 3, 3,  0, 3,-3) :   0.2102610435016800,
   ( 2, 3, 3,  1,-3, 2) :  -0.2102610435016800,   ( 2, 3, 3,  1,-2, 1) :   0.1628675039676400,   ( 2, 3, 3,  1,-1, 0) :  -0.0594708038717590,
   ( 2, 3, 3,  1, 0,-1) :  -0.0594708038717590,   ( 2, 3, 3,  1, 1,-2) :   0.1628675039676400,   ( 2, 3, 3,  1, 2,-3) :  -0.2102610435016800,
   ( 2, 3, 3,  2,-3, 1) :   0.1329807601338109,   ( 2, 3, 3,  2,-2, 0) :  -0.1880631945159188,   ( 2, 3, 3,  2,-1,-1) :   0.2060129077457011,
   ( 2, 3, 3,  2, 0,-2) :  -0.1880631945159188,   ( 2, 3, 3,  2, 1,-3) :   0.1329807601338109,   ( 2, 4, 4, -2,-2, 4) :  -0.1061803092398215,
   ( 2, 4, 4, -2,-1, 3) :   0.1592704638597323,   ( 2, 4, 4, -2, 0, 2) :  -0.1903646150271117,   ( 2, 4, 4, -2, 1, 1) :   0.2006619231289297,
   ( 2, 4, 4, -2, 2, 0) :  -0.1903646150271117,   ( 2, 4, 4, -2, 3,-1) :   0.1592704638597323,   ( 2, 4, 4, -2, 4,-2) :  -0.1061803092398215,
   ( 2, 4, 4, -1,-3, 4) :   0.1986451691985598,   ( 2, 4, 4, -1,-2, 3) :  -0.1877020417299061,   ( 2, 4, 4, -1,-1, 2) :   0.1277004659133599,
   ( 2, 4, 4, -1, 0, 1) :  -0.0448693700612124,   ( 2, 4, 4, -1, 1, 0) :  -0.0448693700612124,   ( 2, 4, 4, -1, 2,-1) :   0.1277004659133599,
   ( 2, 4, 4, -1, 3,-2) :  -0.1877020417299061,   ( 2, 4, 4, -1, 4,-3) :   0.1986451691985598,   ( 2, 4, 4,  0,-4, 4) :  -0.2293756838200146,
   ( 2, 4, 4,  0,-3, 3) :   0.0573439209550036,   ( 2, 4, 4,  0,-2, 2) :   0.0655359096628613,   ( 2, 4, 4,  0,-1, 1) :  -0.1392638080335803,
   ( 2, 4, 4,  0, 0, 0) :   0.1638397741571533,   ( 2, 4, 4,  0, 1,-1) :  -0.1392638080335803,   ( 2, 4, 4,  0, 2,-2) :   0.0655359096628613,
   ( 2, 4, 4,  0, 3,-3) :   0.0573439209550036,   ( 2, 4, 4,  0, 4,-4) :  -0.2293756838200146,   ( 2, 4, 4,  1,-4, 3) :   0.1986451691985598,
   ( 2, 4, 4,  1,-3, 2) :  -0.1877020417299061,   ( 2, 4, 4,  1,-2, 1) :   0.1277004659133599,   ( 2, 4, 4,  1,-1, 0) :  -0.0448693700612124,
   ( 2, 4, 4,  1, 0,-1) :  -0.0448693700612124,   ( 2, 4, 4,  1, 1,-2) :   0.1277004659133599,   ( 2, 4, 4,  1, 2,-3) :  -0.1877020417299061,
   ( 2, 4, 4,  1, 3,-4) :   0.1986451691985598,   ( 2, 4, 4,  2,-4, 2) :  -0.1061803092398215,   ( 2, 4, 4,  2,-3, 1) :   0.1592704638597323,
   ( 2, 4, 4,  2,-2, 0) :  -0.1903646150271117,   ( 2, 4, 4,  2,-1,-1) :   0.2006619231289297,   ( 2, 4, 4,  2, 0,-2) :  -0.1903646150271117,
   ( 2, 4, 4,  2, 1,-3) :   0.1592704638597323,   ( 2, 4, 4,  2, 2,-4) :  -0.1061803092398215,   ( 3, 3, 4, -3,-1, 4) :  -0.1661984725325330,
   ( 3, 3, 4, -3, 0, 3) :   0.2035507268673357,   ( 3, 3, 4, -3, 1, 2) :  -0.1884513542570921,   ( 3, 3, 4, -3, 2, 1) :   0.1404633461902507,
   ( 3, 3, 4, -3, 3, 0) :  -0.0769349432110577,   ( 3, 3, 4, -2,-2, 4) :   0.2145613054278704,   ( 3, 3, 4, -2,-1, 3) :  -0.0959547328555626,
   ( 3, 3, 4, -2, 0, 2) :  -0.0444184101729927,   ( 3, 3, 4, -2, 1, 1) :   0.1450699201459755,   ( 3, 3, 4, -2, 2, 0) :  -0.1795148674924679,
   ( 3, 3, 4, -2, 3,-1) :   0.1404633461902507,   ( 3, 3, 4, -1,-3, 4) :  -0.1661984725325330,   ( 3, 3, 4, -1,-2, 3) :  -0.0959547328555626,
   ( 3, 3, 4, -1,-1, 2) :   0.1621931014684337,   ( 3, 3, 4, -1, 0, 1) :  -0.0993225845992799,   ( 3, 3, 4, -1, 1, 0) :  -0.0256449810703526,
   ( 3, 3, 4, -1, 2,-1) :   0.1450699201459755,   ( 3, 3, 4, -1, 3,-2) :  -0.1884513542570921,   ( 3, 3, 4,  0,-3, 3) :   0.2035507268673357,
   ( 3, 3, 4,  0,-2, 2) :  -0.0444184101729927,   ( 3, 3, 4,  0,-1, 1) :  -0.0993225845992799,   ( 3, 3, 4,  0, 0, 0) :   0.1538698864221154,
   ( 3, 3, 4,  0, 1,-1) :  -0.0993225845992799,   ( 3, 3, 4,  0, 2,-2) :  -0.0444184101729927,   ( 3, 3, 4,  0, 3,-3) :   0.2035507268673357,
   ( 3, 3, 4,  1,-3, 2) :  -0.1884513542570921,   ( 3, 3, 4,  1,-2, 1) :   0.1450699201459755,   ( 3, 3, 4,  1,-1, 0) :  -0.0256449810703526,
   ( 3, 3, 4,  1, 0,-1) :  -0.0993225845992799,   ( 3, 3, 4,  1, 1,-2) :   0.1621931014684337,   ( 3, 3, 4,  1, 2,-3) :  -0.0959547328555626,
   ( 3, 3, 4,  1, 3,-4) :  -0.1661984725325330,   ( 3, 3, 4,  2,-3, 1) :   0.1404633461902507,   ( 3, 3, 4,  2,-2, 0) :  -0.1795148674924679,
   ( 3, 3, 4,  2,-1,-1) :   0.1450699201459755,   ( 3, 3, 4,  2, 0,-2) :  -0.0444184101729927,   ( 3, 3, 4,  2, 1,-3) :  -0.0959547328555626,
   ( 3, 3, 4,  2, 2,-4) :   0.2145613054278704,   ( 3, 3, 4,  3,-3, 0) :  -0.0769349432110577,   ( 3, 3, 4,  3,-2,-1) :   0.1404633461902507,
   ( 3, 3, 4,  3,-1,-2) :  -0.1884513542570921,   ( 3, 3, 4,  3, 0,-3) :   0.2035507268673357,   ( 3, 3, 4,  3, 1,-4) :  -0.1661984725325330,
   ( 4, 4, 4, -4, 0, 4) :   0.1065253059845414,   ( 4, 4, 4, -4, 1, 3) :  -0.1684312976787581,   ( 4, 4, 4, -4, 2, 2) :   0.1909831399962363,
   ( 4, 4, 4, -4, 3, 1) :  -0.1684312976787581,   ( 4, 4, 4, -4, 4, 0) :   0.1065253059845414,   ( 4, 4, 4, -3,-1, 4) :  -0.1684312976787581,
   ( 4, 4, 4, -3, 0, 3) :   0.1597879589768121,   ( 4, 4, 4, -3, 1, 2) :  -0.0636610466654121,   ( 4, 4, 4, -3, 2, 1) :  -0.0636610466654121,
   ( 4, 4, 4, -3, 3, 0) :   0.1597879589768121,   ( 4, 4, 4, -3, 4,-1) :  -0.1684312976787581,   ( 4, 4, 4, -2,-2, 4) :   0.1909831399962363,
   ( 4, 4, 4, -2,-1, 3) :  -0.0636610466654121,   ( 4, 4, 4, -2, 0, 2) :  -0.0836984547021397,   ( 4, 4, 4, -2, 1, 1) :   0.1443696837246498,
   ( 4, 4, 4, -2, 2, 0) :  -0.0836984547021397,   ( 4, 4, 4, -2, 3,-1) :  -0.0636610466654121,   ( 4, 4, 4, -2, 4,-2) :   0.1909831399962363,
   ( 4, 4, 4, -1,-3, 4) :  -0.1684312976787581,   ( 4, 4, 4, -1,-2, 3) :  -0.0636610466654121,   ( 4, 4, 4, -1,-1, 2) :   0.1443696837246498,
   ( 4, 4, 4, -1, 0, 1) :  -0.0684805538472052,   ( 4, 4, 4, -1, 1, 0) :  -0.0684805538472052,   ( 4, 4, 4, -1, 2,-1) :   0.1443696837246498,
   ( 4, 4, 4, -1, 3,-2) :  -0.0636610466654121,   ( 4, 4, 4, -1, 4,-3) :  -0.1684312976787581,   ( 4, 4, 4,  0,-4, 4) :   0.1065253059845414,
   ( 4, 4, 4,  0,-3, 3) :   0.1597879589768121,   ( 4, 4, 4,  0,-2, 2) :  -0.0836984547021397,   ( 4, 4, 4,  0,-1, 1) :  -0.0684805538472052,
   ( 4, 4, 4,  0, 0, 0) :   0.1369611076944104,   ( 4, 4, 4,  0, 1,-1) :  -0.0684805538472052,   ( 4, 4, 4,  0, 2,-2) :  -0.0836984547021397,
   ( 4, 4, 4,  0, 3,-3) :   0.1597879589768121,   ( 4, 4, 4,  0, 4,-4) :   0.1065253059845414,   ( 4, 4, 4,  1,-4, 3) :  -0.1684312976787581,
   ( 4, 4, 4,  1,-3, 2) :  -0.0636610466654121,   ( 4, 4, 4,  1,-2, 1) :   0.1443696837246498,   ( 4, 4, 4,  1,-1, 0) :  -0.0684805538472052,
   ( 4, 4, 4,  1, 0,-1) :  -0.0684805538472052,   ( 4, 4, 4,  1, 1,-2) :   0.1443696837246498,   ( 4, 4, 4,  1, 2,-3) :  -0.0636610466654121,
   ( 4, 4, 4,  1, 3,-4) :  -0.1684312976787581,   ( 4, 4, 4,  2,-4, 2) :   0.1909831399962363,   ( 4, 4, 4,  2,-3, 1) :  -0.0636610466654121,
   ( 4, 4, 4,  2,-2, 0) :  -0.0836984547021397,   ( 4, 4, 4,  2,-1,-1) :   0.1443696837246498,   ( 4, 4, 4,  2, 0,-2) :  -0.0836984547021397,
   ( 4, 4, 4,  2, 1,-3) :  -0.0636610466654121,   ( 4, 4, 4,  2, 2,-4) :   0.1909831399962363,   ( 4, 4, 4,  3,-4, 1) :  -0.1684312976787581,
   ( 4, 4, 4,  3,-3, 0) :   0.1597879589768121,   ( 4, 4, 4,  3,-2,-1) :  -0.0636610466654121,   ( 4, 4, 4,  3,-1,-2) :  -0.0636610466654121,
   ( 4, 4, 4,  3, 0,-3) :   0.1597879589768121,   ( 4, 4, 4,  3, 1,-4) :  -0.1684312976787581,   ( 4, 4, 4,  4,-4, 0) :   0.1065253059845414,
   ( 4, 4, 4,  4,-3,-1) :  -0.1684312976787581,   ( 4, 4, 4,  4,-2,-2) :   0.1909831399962363,   ( 4, 4, 4,  4,-1,-3) :  -0.1684312976787581,
   ( 4, 4, 4,  4, 0,-4) :   0.1065253059845414,}

GAUNT_COEFF_DATA2 = {
   ( 0, 0, 0,  0, 0, 0) :   0.2820947917738781,   ( 0, 1, 1,  0,-1,-1) :   0.2820947917738781,   ( 0, 1, 1,  0, 0, 0) :   0.2820947917738781,
   ( 0, 1, 1,  0, 1, 1) :   0.2820947917738781,   ( 0, 2, 2,  0,-2,-2) :   0.2820947917738781,   ( 0, 2, 2,  0,-1,-1) :   0.2820947917738781,
   ( 0, 2, 2,  0, 0, 0) :   0.2820947917738781,   ( 0, 2, 2,  0, 1, 1) :   0.2820947917738781,   ( 0, 2, 2,  0, 2, 2) :   0.2820947917738781,
   ( 0, 3, 3,  0,-3,-3) :   0.2820947917738781,   ( 0, 3, 3,  0,-2,-2) :   0.2820947917738781,   ( 0, 3, 3,  0,-1,-1) :   0.2820947917738781,
   ( 0, 3, 3,  0, 0, 0) :   0.2820947917738781,   ( 0, 3, 3,  0, 1, 1) :   0.2820947917738781,   ( 0, 3, 3,  0, 2, 2) :   0.2820947917738781,
   ( 0, 3, 3,  0, 3, 3) :   0.2820947917738781,   ( 0, 4, 4,  0,-4,-4) :   0.2820947917738781,   ( 0, 4, 4,  0,-3,-3) :   0.2820947917738781,
   ( 0, 4, 4,  0,-2,-2) :   0.2820947917738781,   ( 0, 4, 4,  0,-1,-1) :   0.2820947917738781,   ( 0, 4, 4,  0, 0, 0) :   0.2820947917738781,
   ( 0, 4, 4,  0, 1, 1) :   0.2820947917738781,   ( 0, 4, 4,  0, 2, 2) :   0.2820947917738781,   ( 0, 4, 4,  0, 3, 3) :   0.2820947917738781,
   ( 0, 4, 4,  0, 4, 4) :   0.2820947917738781,   ( 1, 1, 2, -1,-1, 0) :  -0.1261566261010080,   ( 1, 1, 2, -1,-1, 2) :  -0.2185096861184158,
   ( 1, 1, 2, -1, 0,-1) :   0.2185096861184158,   ( 1, 1, 2, -1, 1,-2) :   0.2185096861184158,   ( 1, 1, 2,  0,-1,-1) :   0.2185096861184158,
   ( 1, 1, 2,  0, 0, 0) :   0.2523132522020160,   ( 1, 1, 2,  0, 1, 1) :   0.2185096861184158,   ( 1, 1, 2,  1,-1,-2) :   0.2185096861184158,
   ( 1, 1, 2,  1, 0, 1) :   0.2185096861184158,   ( 1, 1, 2,  1, 1, 0) :  -0.1261566261010080,   ( 1, 1, 2,  1, 1, 2) :   0.2185096861184158,
   ( 1, 2, 3, -1,-2, 1) :  -0.0583991700819018,   ( 1, 2, 3, -1,-2, 3) :  -0.2261790131595402,   ( 1, 2, 3, -1,-1, 0) :  -0.1430481681026688,
   ( 1, 2, 3, -1,-1, 2) :  -0.1846743909223718,   ( 1, 2, 3, -1, 0,-1) :   0.2023006594034206,   ( 1, 2, 3, -1, 1,-2) :   0.1846743909223718,
   ( 1, 2, 3, -1, 2,-3) :   0.2261790131595402,   ( 1, 2, 3, -1, 2,-1) :   0.0583991700819018,   ( 1, 2, 3,  0,-2,-2) :   0.1846743909223718,
   ( 1, 2, 3,  0,-1,-1) :   0.2335966803276073,   ( 1, 2, 3,  0, 0, 0) :   0.2477666950834761,   ( 1, 2, 3,  0, 1, 1) :   0.2335966803276073,
   ( 1, 2, 3,  0, 2, 2) :   0.1846743909223718,   ( 1, 2, 3,  1,-2,-3) :   0.2261790131595402,   ( 1, 2, 3,  1,-2,-1) :  -0.0583991700819018,
   ( 1, 2, 3,  1,-1,-2) :   0.1846743909223718,   ( 1, 2, 3,  1, 0, 1) :   0.2023006594034206,   ( 1, 2, 3,  1, 1, 0) :  -0.1430481681026688,
   ( 1, 2, 3,  1, 1, 2) :   0.1846743909223718,   ( 1, 2, 3,  1, 2, 1) :  -0.0583991700819018,   ( 1, 2, 3,  1, 2, 3) :   0.2261790131595402,
   ( 1, 3, 4, -1,-3, 2) :  -0.0435281713775682,   ( 1, 3, 4, -1,-3, 4) :  -0.2303294329808903,   ( 1, 3, 4, -1,-2, 1) :  -0.0753930043865134,
   ( 1, 3, 4, -1,-2, 3) :  -0.1994711402007163,   ( 1, 3, 4, -1,-1, 0) :  -0.1507860087730268,   ( 1, 3, 4, -1,-1, 2) :  -0.1685838828361838,
   ( 1, 3, 4, -1, 0,-1) :   0.1946639002730061,   ( 1, 3, 4, -1, 1,-2) :   0.1685838828361838,   ( 1, 3, 4, -1, 2,-3) :   0.1994711402007163,
   ( 1, 3, 4, -1, 2,-1) :   0.0753930043865134,   ( 1, 3, 4, -1, 3,-4) :   0.2303294329808903,   ( 1, 3, 4, -1, 3,-2) :   0.0435281713775682,
   ( 1, 3, 4,  0,-3,-3) :   0.1628675039676399,   ( 1, 3, 4,  0,-2,-2) :   0.2132436186229230,   ( 1, 3, 4,  0,-1,-1) :   0.2384136135044480,
   ( 1, 3, 4,  0, 0, 0) :   0.2462325212298291,   ( 1, 3, 4,  0, 1, 1) :   0.2384136135044480,   ( 1, 3, 4,  0, 2, 2) :   0.2132436186229230,
   ( 1, 3, 4,  0, 3, 3) :   0.1628675039676399,   ( 1, 3, 4,  1,-3,-4) :   0.2303294329808903,   ( 1, 3, 4,  1,-3,-2) :  -0.0435281713775682,
   ( 1, 3, 4,  1,-2,-3) :   0.1994711402007163,   ( 1, 3, 4,  1,-2,-1) :  -0.0753930043865134,   ( 1, 3, 4,  1,-1,-2) :   0.1685838828361838,
   ( 1, 3, 4,  1, 0, 1) :   0.1946639002730061,   ( 1, 3, 4,  1, 1, 0) :  -0.1507860087730268,   ( 1, 3, 4,  1, 1, 2) :   0.1685838828361838,
   ( 1, 3, 4,  1, 2, 1) :  -0.0753930043865134,   ( 1, 3, 4,  1, 2, 3) :   0.1994711402007163,   ( 1, 3, 4,  1, 3, 2) :  -0.0435281713775682,
   ( 1, 3, 4,  1, 3, 4) :   0.2303294329808903,   ( 2, 2, 2, -2,-2, 0) :  -0.1802237515728685,   ( 2, 2, 2, -2,-1, 1) :   0.1560783472274398,
   ( 2, 2, 2, -2, 0,-2) :  -0.1802237515728685,   ( 2, 2, 2, -2, 1,-1) :   0.1560783472274398,   ( 2, 2, 2, -1,-2, 1) :   0.1560783472274398,
   ( 2, 2, 2, -1,-1, 0) :   0.0901118757864343,   ( 2, 2, 2, -1,-1, 2) :  -0.1560783472274398,   ( 2, 2, 2, -1, 0,-1) :   0.0901118757864343,
   ( 2, 2, 2, -1, 1,-2) :   0.1560783472274398,   ( 2, 2, 2, -1, 2,-1) :  -0.1560783472274398,   ( 2, 2, 2,  0,-2,-2) :  -0.1802237515728685,
   ( 2, 2, 2,  0,-1,-1) :   0.0901118757864343,   ( 2, 2, 2,  0, 0, 0) :   0.1802237515728686,   ( 2, 2, 2,  0, 1, 1) :   0.0901118757864343,
   ( 2, 2, 2,  0, 2, 2) :  -0.1802237515728685,   ( 2, 2, 2,  1,-2,-1) :   0.1560783472274398,   ( 2, 2, 2,  1,-1,-2) :   0.1560783472274398,
   ( 2, 2, 2,  1, 0, 1) :   0.0901118757864343,   ( 2, 2, 2,  1, 1, 0) :   0.0901118757864343,   ( 2, 2, 2,  1, 1, 2) :   0.1560783472274398,
   ( 2, 2, 2,  1, 2, 1) :   0.1560783472274398,   ( 2, 2, 2,  2,-1,-1) :  -0.1560783472274398,   ( 2, 2, 2,  2, 0, 2) :  -0.1802237515728685,
   ( 2, 2, 2,  2, 1, 1) :   0.1560783472274398,   ( 2, 2, 2,  2, 2, 0) :  -0.1802237515728685,   ( 2, 2, 4, -2,-2, 0) :   0.0402992559676969,
   ( 2, 2, 4, -2,-2, 4) :  -0.2384136135044480,   ( 2, 2, 4, -2,-1, 1) :  -0.0637187184340275,   ( 2, 2, 4, -2,-1, 3) :  -0.1685838828361838,
   ( 2, 2, 4, -2, 0,-2) :   0.1560783472274399,   ( 2, 2, 4, -2, 1,-3) :   0.1685838828361838,   ( 2, 2, 4, -2, 1,-1) :  -0.0637187184340275,
   ( 2, 2, 4, -2, 2,-4) :   0.2384136135044480,   ( 2, 2, 4, -1,-2, 1) :  -0.0637187184340275,   ( 2, 2, 4, -1,-2, 3) :  -0.1685838828361838,
   ( 2, 2, 4, -1,-1, 0) :  -0.1611970238707875,   ( 2, 2, 4, -1,-1, 2) :  -0.1802237515728685,   ( 2, 2, 4, -1, 0,-1) :   0.2207281154418226,
   ( 2, 2, 4, -1, 1,-2) :   0.1802237515728685,   ( 2, 2, 4, -1, 2,-3) :   0.1685838828361838,   ( 2, 2, 4, -1, 2,-1) :   0.0637187184340275,
   ( 2, 2, 4,  0,-2,-2) :   0.1560783472274399,   ( 2, 2, 4,  0,-1,-1) :   0.2207281154418226,   ( 2, 2, 4,  0, 0, 0) :   0.2417955358061813,
   ( 2, 2, 4,  0, 1, 1) :   0.2207281154418226,   ( 2, 2, 4,  0, 2, 2) :   0.1560783472274399,   ( 2, 2, 4,  1,-2,-3) :   0.1685838828361838,
   ( 2, 2, 4,  1,-2,-1) :  -0.0637187184340275,   ( 2, 2, 4,  1,-1,-2) :   0.1802237515728685,   ( 2, 2, 4,  1, 0, 1) :   0.2207281154418226,
   ( 2, 2, 4,  1, 1, 0) :  -0.1611970238707875,   ( 2, 2, 4,  1, 1, 2) :   0.1802237515728685,   ( 2, 2, 4,  1, 2, 1) :  -0.0637187184340275,
   ( 2, 2, 4,  1, 2, 3) :   0.1685838828361838,   ( 2, 2, 4,  2,-2,-4) :   0.2384136135044480,   ( 2, 2, 4,  2,-1,-3) :   0.1685838828361838,
   ( 2, 2, 4,  2,-1,-1) :   0.0637187184340275,   ( 2, 2, 4,  2, 0, 2) :   0.1560783472274399,   ( 2, 2, 4,  2, 1, 1) :  -0.0637187184340275,
   ( 2, 2, 4,  2, 1, 3) :   0.1685838828361838,   ( 2, 2, 4,  2, 2, 0) :   0.0402992559676969,   ( 2, 2, 4,  2, 2, 4) :   0.2384136135044480,
   ( 2, 3, 3, -2,-3, 1) :  -0.0940315972579594,   ( 2, 3, 3, -2,-2, 0) :  -0.1880631945159187,   ( 2, 3, 3, -2,-1, 1) :   0.1456731240789438,
   ( 2, 3, 3, -2,-1, 3) :   0.0940315972579594,   ( 2, 3, 3, -2, 0,-2) :  -0.1880631945159187,   ( 2, 3, 3, -2, 1,-3) :  -0.0940315972579594,
   ( 2, 3, 3, -2, 1,-1) :   0.1456731240789438,   ( 2, 3, 3, -2, 3,-1) :   0.0940315972579594,   ( 2, 3, 3, -1,-3, 2) :   0.1486770096793976,
   ( 2, 3, 3, -1,-2, 1) :   0.1151647164904451,   ( 2, 3, 3, -1,-2, 3) :  -0.1486770096793976,   ( 2, 3, 3, -1,-1, 0) :   0.0594708038717590,
   ( 2, 3, 3, -1,-1, 2) :  -0.1151647164904451,   ( 2, 3, 3, -1, 0,-1) :   0.0594708038717590,   ( 2, 3, 3, -1, 1,-2) :   0.1151647164904451,
   ( 2, 3, 3, -1, 2,-3) :   0.1486770096793976,   ( 2, 3, 3, -1, 2,-1) :  -0.1151647164904451,   ( 2, 3, 3, -1, 3,-2) :  -0.1486770096793976,
   ( 2, 3, 3,  0,-3,-3) :  -0.2102610435016800,   ( 2, 3, 3,  0,-1,-1) :   0.1261566261010080,   ( 2, 3, 3,  0, 0, 0) :   0.1682088348013440,
   ( 2, 3, 3,  0, 1, 1) :   0.1261566261010080,   ( 2, 3, 3,  0, 3, 3) :  -0.2102610435016800,   ( 2, 3, 3,  1,-3,-2) :   0.1486770096793976,
   ( 2, 3, 3,  1,-2,-3) :   0.1486770096793976,   ( 2, 3, 3,  1,-2,-1) :   0.1151647164904451,   ( 2, 3, 3,  1,-1,-2) :   0.1151647164904451,
   ( 2, 3, 3,  1, 0, 1) :   0.0594708038717590,   ( 2, 3, 3,  1, 1, 0) :   0.0594708038717590,   ( 2, 3, 3,  1, 1, 2) :   0.1151647164904451,
   ( 2, 3, 3,  1, 2, 1) :   0.1151647164904451,   ( 2, 3, 3,  1, 2, 3) :   0.1486770096793976,   ( 2, 3, 3,  1, 3, 2) :   0.1486770096793976,
   ( 2, 3, 3,  2,-3,-1) :  -0.0940315972579594,   ( 2, 3, 3,  2,-1,-3) :  -0.0940315972579594,   ( 2, 3, 3,  2,-1,-1) :  -0.1456731240789438,
   ( 2, 3, 3,  2, 0, 2) :  -0.1880631945159187,   ( 2, 3, 3,  2, 1, 1) :   0.1456731240789438,   ( 2, 3, 3,  2, 1, 3) :  -0.0940315972579594,
   ( 2, 3, 3,  2, 2, 0) :  -0.1880631945159187,   ( 2, 3, 3,  2, 3, 1) :  -0.0940315972579594,   ( 2, 4, 4, -2,-4, 2) :  -0.0750808166919624,
   ( 2, 4, 4, -2,-3, 1) :  -0.1126212250379436,   ( 2, 4, 4, -2,-2, 0) :  -0.1903646150271116,   ( 2, 4, 4, -2,-2, 4) :   0.0750808166919624,
   ( 2, 4, 4, -2,-1, 1) :   0.1418894065703999,   ( 2, 4, 4, -2,-1, 3) :   0.1126212250379436,   ( 2, 4, 4, -2, 0,-2) :  -0.1903646150271116,
   ( 2, 4, 4, -2, 1,-3) :  -0.1126212250379436,   ( 2, 4, 4, -2, 1,-1) :   0.1418894065703999,   ( 2, 4, 4, -2, 2,-4) :  -0.0750808166919624,
   ( 2, 4, 4, -2, 3,-1) :   0.1126212250379436,   ( 2, 4, 4, -2, 4,-2) :   0.0750808166919624,   ( 2, 4, 4, -1,-4, 3) :   0.1404633461902507,
   ( 2, 4, 4, -1,-3, 2) :   0.1327253865497769,   ( 2, 4, 4, -1,-3, 4) :  -0.1404633461902507,   ( 2, 4, 4, -1,-2, 1) :   0.0902978654080183,
   ( 2, 4, 4, -1,-2, 3) :  -0.1327253865497769,   ( 2, 4, 4, -1,-1, 0) :   0.0448693700612124,   ( 2, 4, 4, -1,-1, 2) :  -0.0902978654080183,
   ( 2, 4, 4, -1, 0,-1) :   0.0448693700612124,   ( 2, 4, 4, -1, 1,-2) :   0.0902978654080183,   ( 2, 4, 4, -1, 2,-3) :   0.1327253865497769,
   ( 2, 4, 4, -1, 2,-1) :  -0.0902978654080183,   ( 2, 4, 4, -1, 3,-4) :   0.1404633461902507,   ( 2, 4, 4, -1, 3,-2) :  -0.1327253865497769,
   ( 2, 4, 4, -1, 4,-3) :  -0.1404633461902507,   ( 2, 4, 4,  0,-4,-4) :  -0.2293756838200145,   ( 2, 4, 4,  0,-3,-3) :  -0.0573439209550036,
   ( 2, 4, 4,  0,-2,-2) :   0.0655359096628613,   ( 2, 4, 4,  0,-1,-1) :   0.1392638080335802,   ( 2, 4, 4,  0, 0, 0) :   0.1638397741571533,
   ( 2, 4, 4,  0, 1, 1) :   0.1392638080335802,   ( 2, 4, 4,  0, 2, 2) :   0.0655359096628613,   ( 2, 4, 4,  0, 3, 3) :  -0.0573439209550036,
   ( 2, 4, 4,  0, 4, 4) :  -0.2293756838200145,   ( 2, 4, 4,  1,-4,-3) :   0.1404633461902507,   ( 2, 4, 4,  1,-3,-4) :   0.1404633461902507,
   ( 2, 4, 4,  1,-3,-2) :   0.1327253865497769,   ( 2, 4, 4,  1,-2,-3) :   0.1327253865497769,   ( 2, 4, 4,  1,-2,-1) :   0.0902978654080183,
   ( 2, 4, 4,  1,-1,-2) :   0.0902978654080183,   ( 2, 4, 4,  1, 0, 1) :   0.0448693700612124,   ( 2, 4, 4,  1, 1, 0) :   0.0448693700612124,
   ( 2, 4, 4,  1, 1, 2) :   0.0902978654080183,   ( 2, 4, 4,  1, 2, 1) :   0.0902978654080183,   ( 2, 4, 4,  1, 2, 3) :   0.1327253865497769,
   ( 2, 4, 4,  1, 3, 2) :   0.1327253865497769,   ( 2, 4, 4,  1, 3, 4) :   0.1404633461902507,   ( 2, 4, 4,  1, 4, 3) :   0.1404633461902507,
   ( 2, 4, 4,  2,-4,-2) :  -0.0750808166919624,   ( 2, 4, 4,  2,-3,-1) :  -0.1126212250379436,   ( 2, 4, 4,  2,-2,-4) :  -0.0750808166919624,
   ( 2, 4, 4,  2,-1,-3) :  -0.1126212250379436,   ( 2, 4, 4,  2,-1,-1) :  -0.1418894065703999,   ( 2, 4, 4,  2, 0, 2) :  -0.1903646150271116,
   ( 2, 4, 4,  2, 1, 1) :   0.1418894065703999,   ( 2, 4, 4,  2, 1, 3) :  -0.1126212250379436,   ( 2, 4, 4,  2, 2, 0) :  -0.1903646150271116,
   ( 2, 4, 4,  2, 2, 4) :  -0.0750808166919624,   ( 2, 4, 4,  2, 3, 1) :  -0.1126212250379436,   ( 2, 4, 4,  2, 4, 2) :  -0.0750808166919624,
   ( 3, 3, 4, -3,-3, 0) :   0.0769349432110577,   ( 3, 3, 4, -3,-2, 1) :  -0.0993225845992799,   ( 3, 3, 4, -3,-1, 2) :   0.1332552305189781,
   ( 3, 3, 4, -3,-1, 4) :   0.1175200669506002,   ( 3, 3, 4, -3, 0,-3) :  -0.2035507268673356,   ( 3, 3, 4, -3, 1,-4) :  -0.1175200669506002,
   ( 3, 3, 4, -3, 1,-2) :   0.1332552305189781,   ( 3, 3, 4, -3, 2,-1) :  -0.0993225845992799,   ( 3, 3, 4, -2,-3, 1) :  -0.0993225845992799,
   ( 3, 3, 4, -2,-2, 0) :  -0.1795148674924679,   ( 3, 3, 4, -2,-2, 4) :  -0.1517177540482851,   ( 3, 3, 4, -2,-1, 1) :   0.1025799242814102,
   ( 3, 3, 4, -2,-1, 3) :  -0.0678502422891119,   ( 3, 3, 4, -2, 0,-2) :  -0.0444184101729927,   ( 3, 3, 4, -2, 1,-3) :   0.0678502422891119,
   ( 3, 3, 4, -2, 1,-1) :   0.1025799242814102,   ( 3, 3, 4, -2, 2,-4) :   0.1517177540482851,   ( 3, 3, 4, -2, 3,-1) :   0.0993225845992799,
   ( 3, 3, 4, -1,-3, 2) :   0.1332552305189781,   ( 3, 3, 4, -1,-3, 4) :   0.1175200669506002,   ( 3, 3, 4, -1,-2, 1) :   0.1025799242814102,
   ( 3, 3, 4, -1,-2, 3) :  -0.0678502422891119,   ( 3, 3, 4, -1,-1, 0) :   0.0256449810703526,   ( 3, 3, 4, -1,-1, 2) :  -0.1146878419100072,
   ( 3, 3, 4, -1, 0,-1) :   0.0993225845992799,   ( 3, 3, 4, -1, 1,-2) :   0.1146878419100072,   ( 3, 3, 4, -1, 2,-3) :   0.0678502422891119,
   ( 3, 3, 4, -1, 2,-1) :  -0.1025799242814102,   ( 3, 3, 4, -1, 3,-4) :  -0.1175200669506002,   ( 3, 3, 4, -1, 3,-2) :  -0.1332552305189781,
   ( 3, 3, 4,  0,-3,-3) :  -0.2035507268673356,   ( 3, 3, 4,  0,-2,-2) :  -0.0444184101729927,   ( 3, 3, 4,  0,-1,-1) :   0.0993225845992799,
   ( 3, 3, 4,  0, 0, 0) :   0.1538698864221154,   ( 3, 3, 4,  0, 1, 1) :   0.0993225845992799,   ( 3, 3, 4,  0, 2, 2) :  -0.0444184101729927,
   ( 3, 3, 4,  0, 3, 3) :  -0.2035507268673356,   ( 3, 3, 4,  1,-3,-4) :  -0.1175200669506002,   ( 3, 3, 4,  1,-3,-2) :   0.1332552305189781,
   ( 3, 3, 4,  1,-2,-3) :   0.0678502422891119,   ( 3, 3, 4,  1,-2,-1) :   0.1025799242814102,   ( 3, 3, 4,  1,-1,-2) :   0.1146878419100072,
   ( 3, 3, 4,  1, 0, 1) :   0.0993225845992799,   ( 3, 3, 4,  1, 1, 0) :   0.0256449810703526,   ( 3, 3, 4,  1, 1, 2) :   0.1146878419100072,
   ( 3, 3, 4,  1, 2, 1) :   0.1025799242814102,   ( 3, 3, 4,  1, 2, 3) :   0.0678502422891119,   ( 3, 3, 4,  1, 3, 2) :   0.1332552305189781,
   ( 3, 3, 4,  1, 3, 4) :  -0.1175200669506002,   ( 3, 3, 4,  2,-3,-1) :  -0.0993225845992799,   ( 3, 3, 4,  2,-2,-4) :   0.1517177540482851,
   ( 3, 3, 4,  2,-1,-3) :   0.0678502422891119,   ( 3, 3, 4,  2,-1,-1) :  -0.1025799242814102,   ( 3, 3, 4,  2, 0, 2) :  -0.0444184101729927,
   ( 3, 3, 4,  2, 1, 1) :   0.1025799242814102,   ( 3, 3, 4,  2, 1, 3) :   0.0678502422891119,   ( 3, 3, 4,  2, 2, 0) :  -0.1795148674924679,
   ( 3, 3, 4,  2, 2, 4) :   0.1517177540482851,   ( 3, 3, 4,  2, 3, 1) :  -0.0993225845992799,   ( 3, 3, 4,  3,-2,-1) :   0.0993225845992799,
   ( 3, 3, 4,  3,-1,-4) :  -0.1175200669506002,   ( 3, 3, 4,  3,-1,-2) :  -0.1332552305189781,   ( 3, 3, 4,  3, 0, 3) :  -0.2035507268673356,
   ( 3, 3, 4,  3, 1, 2) :   0.1332552305189781,   ( 3, 3, 4,  3, 1, 4) :  -0.1175200669506002,   ( 3, 3, 4,  3, 2, 1) :  -0.0993225845992799,
   ( 3, 3, 4,  3, 3, 0) :   0.0769349432110577,   ( 4, 4, 4, -4,-4, 0) :   0.1065253059845414,   ( 4, 4, 4, -4,-3, 1) :  -0.1190989127526998,
   ( 4, 4, 4, -4,-2, 2) :   0.1350454733836384,   ( 4, 4, 4, -4,-1, 3) :  -0.1190989127526998,   ( 4, 4, 4, -4, 0,-4) :   0.1065253059845414,
   ( 4, 4, 4, -4, 1,-3) :  -0.1190989127526998,   ( 4, 4, 4, -4, 2,-2) :   0.1350454733836384,   ( 4, 4, 4, -4, 3,-1) :  -0.1190989127526998,
   ( 4, 4, 4, -3,-4, 1) :  -0.1190989127526998,   ( 4, 4, 4, -3,-3, 0) :  -0.1597879589768121,   ( 4, 4, 4, -3,-2, 1) :   0.0450151577945461,
   ( 4, 4, 4, -3,-1, 2) :   0.0450151577945461,   ( 4, 4, 4, -3,-1, 4) :   0.1190989127526998,   ( 4, 4, 4, -3, 0,-3) :  -0.1597879589768121,
   ( 4, 4, 4, -3, 1,-4) :  -0.1190989127526998,   ( 4, 4, 4, -3, 1,-2) :   0.0450151577945461,   ( 4, 4, 4, -3, 2,-1) :   0.0450151577945461,
   ( 4, 4, 4, -3, 4,-1) :   0.1190989127526998,   ( 4, 4, 4, -2,-4, 2) :   0.1350454733836384,   ( 4, 4, 4, -2,-3, 1) :   0.0450151577945461,
   ( 4, 4, 4, -2,-2, 0) :  -0.0836984547021397,   ( 4, 4, 4, -2,-2, 4) :  -0.1350454733836384,   ( 4, 4, 4, -2,-1, 1) :   0.1020847823594570,
   ( 4, 4, 4, -2,-1, 3) :  -0.0450151577945461,   ( 4, 4, 4, -2, 0,-2) :  -0.0836984547021397,   ( 4, 4, 4, -2, 1,-3) :   0.0450151577945461,
   ( 4, 4, 4, -2, 1,-1) :   0.1020847823594570,   ( 4, 4, 4, -2, 2,-4) :   0.1350454733836384,   ( 4, 4, 4, -2, 3,-1) :  -0.0450151577945461,
   ( 4, 4, 4, -2, 4,-2) :  -0.1350454733836384,   ( 4, 4, 4, -1,-4, 3) :  -0.1190989127526998,   ( 4, 4, 4, -1,-3, 2) :   0.0450151577945461,
   ( 4, 4, 4, -1,-3, 4) :   0.1190989127526998,   ( 4, 4, 4, -1,-2, 1) :   0.1020847823594570,   ( 4, 4, 4, -1,-2, 3) :  -0.0450151577945461,
   ( 4, 4, 4, -1,-1, 0) :   0.0684805538472052,   ( 4, 4, 4, -1,-1, 2) :  -0.1020847823594570,   ( 4, 4, 4, -1, 0,-1) :   0.0684805538472052,
   ( 4, 4, 4, -1, 1,-2) :   0.1020847823594570,   ( 4, 4, 4, -1, 2,-3) :   0.0450151577945461,   ( 4, 4, 4, -1, 2,-1) :  -0.1020847823594570,
   ( 4, 4, 4, -1, 3,-4) :  -0.1190989127526998,   ( 4, 4, 4, -1, 3,-2) :  -0.0450151577945461,   ( 4, 4, 4, -1, 4,-3) :   0.1190989127526998,
   ( 4, 4, 4,  0,-4,-4) :   0.1065253059845414,   ( 4, 4, 4,  0,-3,-3) :  -0.1597879589768121,   ( 4, 4, 4,  0,-2,-2) :  -0.0836984547021397,
   ( 4, 4, 4,  0,-1,-1) :   0.0684805538472052,   ( 4, 4, 4,  0, 0, 0) :   0.1369611076944104,   ( 4, 4, 4,  0, 1, 1) :   0.0684805538472052,
   ( 4, 4, 4,  0, 2, 2) :  -0.0836984547021397,   ( 4, 4, 4,  0, 3, 3) :  -0.1597879589768121,   ( 4, 4, 4,  0, 4, 4) :   0.1065253059845414,
   ( 4, 4, 4,  1,-4,-3) :  -0.1190989127526998,   ( 4, 4, 4,  1,-3,-4) :  -0.1190989127526998,   ( 4, 4, 4,  1,-3,-2) :   0.0450151577945461,
   ( 4, 4, 4,  1,-2,-3) :   0.0450151577945461,   ( 4, 4, 4,  1,-2,-1) :   0.1020847823594570,   ( 4, 4, 4,  1,-1,-2) :   0.1020847823594570,
   ( 4, 4, 4,  1, 0, 1) :   0.0684805538472052,   ( 4, 4, 4,  1, 1, 0) :   0.0684805538472052,   ( 4, 4, 4,  1, 1, 2) :   0.1020847823594570,
   ( 4, 4, 4,  1, 2, 1) :   0.1020847823594570,   ( 4, 4, 4,  1, 2, 3) :   0.0450151577945461,   ( 4, 4, 4,  1, 3, 2) :   0.0450151577945461,
   ( 4, 4, 4,  1, 3, 4) :  -0.1190989127526998,   ( 4, 4, 4,  1, 4, 3) :  -0.1190989127526998,   ( 4, 4, 4,  2,-4,-2) :   0.1350454733836384,
   ( 4, 4, 4,  2,-3,-1) :   0.0450151577945461,   ( 4, 4, 4,  2,-2,-4) :   0.1350454733836384,   ( 4, 4, 4,  2,-1,-3) :   0.0450151577945461,
   ( 4, 4, 4,  2,-1,-1) :  -0.1020847823594570,   ( 4, 4, 4,  2, 0, 2) :  -0.0836984547021397,   ( 4, 4, 4,  2, 1, 1) :   0.1020847823594570,
   ( 4, 4, 4,  2, 1, 3) :   0.0450151577945461,   ( 4, 4, 4,  2, 2, 0) :  -0.0836984547021397,   ( 4, 4, 4,  2, 2, 4) :   0.1350454733836384,
   ( 4, 4, 4,  2, 3, 1) :   0.0450151577945461,   ( 4, 4, 4,  2, 4, 2) :   0.1350454733836384,   ( 4, 4, 4,  3,-4,-1) :  -0.1190989127526998,
   ( 4, 4, 4,  3,-2,-1) :  -0.0450151577945461,   ( 4, 4, 4,  3,-1,-4) :  -0.1190989127526998,   ( 4, 4, 4,  3,-1,-2) :  -0.0450151577945461,
   ( 4, 4, 4,  3, 0, 3) :  -0.1597879589768121,   ( 4, 4, 4,  3, 1, 2) :   0.0450151577945461,   ( 4, 4, 4,  3, 1, 4) :  -0.1190989127526998,
   ( 4, 4, 4,  3, 2, 1) :   0.0450151577945461,   ( 4, 4, 4,  3, 3, 0) :  -0.1597879589768121,   ( 4, 4, 4,  3, 4, 1) :  -0.1190989127526998,
   ( 4, 4, 4,  4,-3,-1) :   0.1190989127526998,   ( 4, 4, 4,  4,-2,-2) :  -0.1350454733836384,   ( 4, 4, 4,  4,-1,-3) :   0.1190989127526998,
   ( 4, 4, 4,  4, 0, 4) :   0.1065253059845414,   ( 4, 4, 4,  4, 1, 3) :  -0.1190989127526998,   ( 4, 4, 4,  4, 2, 2) :   0.1350454733836384,
   ( 4, 4, 4,  4, 3, 1) :  -0.1190989127526998,   ( 4, 4, 4,  4, 4, 0) :   0.1065253059845414,}


def GauntTable(l1=0, l2=0, l3=0, m1=0, m2=0, m3=0, real=True):
    """
    Get Gaunt coefficients from the pre-calculated table.

    If "real = True", the the Gaunt coefficients are defined as the integral
    over three *real* spherical harmonics.
    """

    lmax = np.max([l1, l2, l3])
    assert lmax < GAUNT_LMAX

    # invariant under any permutation
    l_sort_ind = np.argsort([l1, l2, l3])
    l4, l5, l6 = np.array([l1, l2, l3])[l_sort_ind]
    m4, m5, m6 = np.array([m1, m2, m3])[l_sort_ind]
    k = (l4, l5, l6, m4, m5, m6)

    if real:
        g = 0.0 if k not in GAUNT_COEFF_DATA2 else GAUNT_COEFF_DATA2[k]
    else:
        g = 0.0 if k not in GAUNT_COEFF_DATA1 else GAUNT_COEFF_DATA1[k]

    return g


if __name__ == "__main__":
    for k, v in GAUNT_COEFF_DATA2.items():
        # l1, l2, l3, m1, m2, m3 = np.fromstring(k[1:-1], sep=',')
        l1, l2, l3, m1, m2, m3 = k
        # if (l1 + l2 + l3) % 2 != 0 or (m1 + m2 + m3) != 0:
        if (l1 + l2 + l3) % 2 != 0:
            print(k)
        a, b, c = sorted([l1, l2, l3])
        if a + b < c:
            print(k)
    print(
        GauntTable(4, 3, 1, 2, 3, 1),
        GauntTable(3, 1, 4, 3, 1, 2),
        GauntTable(1, 4, 3, 1, 2, 3),
        GauntTable(3, 4, 1, 3, 2, 1),
        GauntTable(4, 1, 3, 2, 1, 3),
        GauntTable(1, 3, 4, 1, 3, 2),
    )
