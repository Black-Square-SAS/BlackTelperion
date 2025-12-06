"""
This package contains reference features such as typical mineral absorbtions or absorbtion positions of QAQC targets.
"""
from BlackTelperion.blackfeature import *

##############################
##Create basic/common features
##############################
#N.B. All values here are approximate and largely for (1) plotting and (2) as initial values for feature fitting
#     routines. Believe nothing!
class Features:
    """
    Specific absorption types. Useful for plotting etc. Not really used for anything and will probably be depreciated soon.
    """

    H2O = [ BlackFeature("H2O", p, w, color='skyblue') for p,w in [(825,50), (940,75), (1130,100), (1400,150), (1900,200), (2700,150)] ]
    OH = [BlackFeature("OH", 1400, 50, color='aquamarine'), BlackFeature("OH", 1550, 50, color='aquamarine'), BlackFeature("OH", 1800, 100, color='aquamarine')]
    AlOH = [BlackFeature("AlOH", 2190, 60, color='salmon')]
    FeOH = [BlackFeature("FeOH", 2265, 70, color='orange')]
    MgOH = [BlackFeature("MgOH", 2330, 60, color='blue'), BlackFeature("MgOH", 2385, 30,color='blue')]
    MgCO3 = [BlackFeature("MgCO3", 2320, 20, color='green')]
    CaCO3 = [BlackFeature("CaCO3", 2340, 20, color='blue')]
    FeCO3 = [BlackFeature("FeCO3", 2350, 20, color='steelblue')]
    Ferrous = [BlackFeature("Fe2+", 1000, 400, color='green')]
    Ferric = [BlackFeature("Fe3+", 650, 170, color='green')]

    # REE features
    Pr = [BlackFeature("Pr", w, 5, color=(74 / 256., 155 / 256., 122 / 256., 1)) for w in [457, 485, 473, 595, 1017] ]
    Nd = [BlackFeature("Nd", w, 5, color=(116 / 256., 114 / 256., 174 / 256., 1)) for w in [430, 463, 475, 514, 525, 580, 627, 680, 750, 800, 880, 1430, 1720, 2335, 2470]]
    Sm = [BlackFeature("Sm", w, 5, color=(116 / 256., 114 / 256., 174 / 256., 1)) for w in [945, 959, 1085, 1235, 1257, 1400, 1550]]
    Eu = [BlackFeature("Eu", w, 5, color=(213 / 256., 64 / 256., 136 / 256., 1)) for w in [385, 405, 470, 530, 1900, 2025, 2170, 2400, 2610]]
    Dy = [BlackFeature("Dy", w, 5, color=(117 / 256., 163 / 256., 58 / 256., 1)) for w in [368, 390, 403, 430, 452, 461, 475, 760, 810, 830, 915, 1117, 1276, 1725]]
    Ho = [BlackFeature("Ho", w, 5, color=(222 / 256., 172 / 256., 59 / 256., 1)) for w in [363, 420, 458, 545, 650, 900, 1130, 1180, 1870, 1930, 2005]]
    Er = [BlackFeature("Er", w, 5, color=(159 / 256., 119 / 256., 49 / 256., 1)) for w in [390, 405, 455, 490, 522, 540, 652, 805, 985, 1485, 1545]]
    Tm = [BlackFeature("Tm", w, 5, color=(102 / 256., 102 / 256., 102 / 256., 1)) for w in [390, 470, 660, 685, 780, 1190, 1640, 1750]]
    Yb = [BlackFeature("Yb", w, 5, color=(209 / 256., 53 / 256., 43 / 256., 1)) for w in [955, 975, 1004 ]]

# common minerals
class Minerals:
    """
    Common mineral absorption features. Useful for plotting etc. Not really used for anything and will probably be depreciated soon.
    """

    # Kaolin clays (dominant SWIR feature)
    Kaolinite = [BlackFeature("Kaolinite/Halloysite", 2200, 100, color='aquamarine')]
    Halloysite = [Kaolinite]
    Dickite = [BlackFeature("Dickite/Nacrite", 2180, 100, color='aquamarine')]
    Nacrite = [Dickite]
    KAOLIN = MultiFeature("Kaolin", Kaolinite + Dickite)

    Pyrophyllite = BlackFeature("Pyrophyllite", 2160.0, 150, color='aquamarine')

    #Smectite clays (dominant SWIR feature)
    Montmorillonite = [BlackFeature("Montmorillonite", 2210.0, 125, color='orange')]
    Nontronite = [BlackFeature("Nontronite", 2280, 125, color='orange')]
    Saponite = [BlackFeature("Saponite", 2309, 100, color='orange')]
    SMECTITE = MultiFeature("Smectite", Montmorillonite + Nontronite + Saponite)

    # white micas (dominant SWIR feature)
    Mica_Na = [BlackFeature("Mica (Na)", 2150, 150, color='coral' )]
    Mica_K = [BlackFeature("Mica (K)", 2190, 150, color='lightcoral' )]
    Mica_MgFe = [BlackFeature("Mica (Mg, Fe)", 2225, 150 , color='sandybrown')]
    MICA = MultiFeature("White mica", Mica_Na + Mica_K +Mica_MgFe)

    # chlorite
    Chlorite_Mg = [ BlackFeature("Chlorite (Mg)", 2245.0, 50, color='seagreen'), BlackFeature("Chlorite (Mg)", 2325.0, 50, color='seagreen') ]
    Chlorite_Fe = [BlackFeature("Chlorite (Fe)", 2261.0, 50, color='seagreen'), BlackFeature("Chlorite (Fe)", 2355.0, 50, color='seagreen') ]
    CHLORITE = [MultiFeature("Chlorite (FeOH)", [Chlorite_Mg[0], Chlorite_Fe[0]]),
                MultiFeature("Chlorite (MgOH)", [Chlorite_Mg[1], Chlorite_Fe[1]])]

    # biotite
    Biotite_Mg = [ BlackFeature("Biotite (Mg)", 2326, 50, color='firebrick'), BlackFeature("Biotite (Mg)", 2377, 50, color='firebrick') ]
    Biotite_Fe = [ BlackFeature("Biotite (Fe)", 2250, 50, color='firebrick'), BlackFeature("Biotite (Fe)", 2350, 50, color='firebrick') ]
    BIOTITE = [MultiFeature("Biotite (FeOH)", [Biotite_Mg[0], Biotite_Fe[0]]),
               MultiFeature("Biotite (MgOH)", [Biotite_Mg[1], Biotite_Fe[1]]) ]

    # amphiboles Tremolite, hornblende, actinolite
    Amphibole_Mg =  [BlackFeature("Amphibole (Mg)", 2320.0, 50, color='royalblue')]
    Amphibole_Fe =  [BlackFeature("Amphibole (Fe)", 2345.0, 50, color='royalblue')]
    AMPHIBOLE = MultiFeature("Amphibole", Amphibole_Mg + Amphibole_Fe)

    # carbonate minerals
    Dolomite = [BlackFeature("Dolomite", 2320, 20, color='green')]
    Calcite = [BlackFeature("Calcite", 2345, 20, color='blue')]
    Ankerite = [BlackFeature("Ankerite", 2330, 20, color='steelblue')]
    CARBONATE = MultiFeature("Carbonate", Dolomite+ Ankerite+ Calcite)

    #Sulphates Jarosite
    Gypsum = [BlackFeature("Gypsum", 1449.0, 50, color='gold'), BlackFeature("Gypsum", 1750, 50, color='gold'), BlackFeature("Gypsum", 1948.0, 50, color='gold')]
    Jarosite = [BlackFeature("Jarosite", 1470.0, 50, color='orange'), BlackFeature("Jarosite", 1850, 50, color='orange'), BlackFeature("Jarosite", 2270.0, 50, color='orange')]
    SULPHATE = MultiFeature( "Sulphate", Gypsum + Jarosite )

    # misc
    Epidote = [ BlackFeature("Epidote", 2256.0, 40, color='green'), BlackFeature("Epidote", 2340.0, 40, color='green')]

# and some useful 'themes' (for plotting etc)
class Themes:
    """
    Some useful 'themes' (for plotting etc)
    """
    ATMOSPHERE = Features.H2O  #[BlackFeature("H2O", 975, 30), BlackFeature("H2O", 1395, 120), BlackFeature("H2O", 1885, 180), BlackFeature("H2O", 2450, 100)]
    CARBONATE = [Minerals.CARBONATE]
    OH = Features.AlOH + Features.FeOH + Features.MgOH
    CLAY = [ Minerals.KAOLIN, Minerals.SMECTITE ]
    DIAGNOSTIC = Features.Ferrous + Features.AlOH+Features.FeOH+Features.MgOH

#expose through BlackFeature class for convenience
BlackFeature.Features = Features
BlackFeature.Minerals = Minerals
BlackFeature.Themes = Themes
