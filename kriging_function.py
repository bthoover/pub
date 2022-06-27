###########################################################################################
#
# KRIG_FLD , KRIG_FLD_05 , KRIG_FLD_95 , KRIG_VAR , HAN1_APCP , HAN05_APCP , HAN95_APCP \
#    = KRIG_INNOVATIONS(
#                        STAT_PRCP_QCd ,
#                        BKG_PRCP_QCd  ,
#                        STAT_LAT_QCd  ,
#                        STAT_LON_QCd  ,
#                        GRID_LATS     ,
#                        GRID_LONS     ,
#                        GRID_BKG_APCP ,
#                        nx            ,
#                        ny            ,
#                        INFL_FACT
#                      )
#
# Computes innovations (obs-minus-background) and krigs them onto the grid specified by
# GRID_LATS, GRID_LONS. GRID_BKG_APCP is used as an external drift variable. The variance
# provided by kriging is inflated by a multiplicative INFL_FACT, which is used to compute
# a 5% and 95% confidence on the kriged field and by-extension a 5% and 95% confidence on
# the QPE analysis.
#
# Use R/GSTAT to perform variogram modeling and kriging. This requires producing R
# Spatial Dataframes for input/output to R routines, and converting the dataframes back
# into numpy arrays.
#
# In this formulation, we replace the ordinary kriging method (stationary mean,
# kriged field reverts to mean value if < 15 observations nearby) with a 'universal'
# kriging, or kriging-with-external-drift (KED) method (mean is assumed to be a linear
# function of predictor variables available on the entire grid, no minimum observation
# threshold). We are utilizing the following predictors:
#    1) model background precip value
#
#    INPUTS:
#        STAT_PRCP_QCd ..................................... Precipitation of all post-QC observations
#        BKG_PRCP_QCd ...................................... Background QPF at all post-QC observations
#        STAT_LAT_QCd ...................................... Latitudes of all post-QC observations
#        STAT_LON_QCd ...................................... Longitudes of all post-QC observations
#        GRID_LATS ......................................... Kriged grid latitudes (2-d array, curvilinear grid)
#        GRID_LONS ......................................... Kriged grid longitudes (2-d array, curvilinear grid)
#        GRID_BKG_APCP ..................................... Background QPF on kriged grid
#        nx ................................................ Number of x-pts on kriged grid
#        ny ................................................ Number of y-pts on kriged grid
#        INFL_FACT ......................................... Multiplicative inflation-factor for variance
#
# OUPUTS:
#        KRIG_FLD .......................................... Innovations on kriged grid (i.e. increment, best-guess)
#        KRIG_FLD_05 ....................................... 5% confidence bound on KRIG_FLD
#        KRIG_FLD_95 ....................................... 95% confidence bound on KRIG_FLD
#        KRIG_VAR .......................................... Variance of KRIG_FLD (representative of interpolative error)
#        HAN1_APCP ......................................... QPE analysis (best-guess)
#        HAN05_APCP ........................................ 5% confidence bound on HAN1_APCP
#        HAN95_APCP ........................................ 95% confidence bound on HAN1_APCP
#
# NOTES:
#
# Uses numpy, rpy2, scipy.stats/norm, R libraries: base, utils, sp, gstat
#
def KRIG_INNOVATIONS(
                      STAT_PRCP_QCd ,
                      BKG_PRCP_QCd  ,
                      STAT_LAT_QCd  ,
                      STAT_LON_QCd  ,
                      GRID_LATS     , # HRRR_LATS_LR
                      GRID_LONS     , # HRRR_LONS_LR
                      GRID_BKG_APCP , # BKG_APCP_LR
                      nx            , # nx_lr
                      ny            , # ny_lr
                      INFL_FACT
                    ):
    # Import modules
    import numpy as np #................................................................... Math and Arrays Module
    from scipy.stats import norm #......................................................... Normal distribution PDF/CDF and inversion module
    import rpy2 #.......................................................................... Pyton/R interface module
    from rpy2.robjects.packages import importr #........................................... R library importer module
    #
    # Import R libraries
    #
    base  = importr('base') #.............................................................. R base library
    utils = importr('utils') #............................................................. R utility library
    sp    = importr('sp') #................................................................ R spatial data library
    gstat = importr('gstat') #............................................................. R gstat library
    #
    # Define input R data
    #
    r_obs_vec = rpy2.robjects.FloatVector(STAT_PRCP_QCd - BKG_PRCP_QCd) #.................. Vector of kriging observations
    r_bkg_vec = rpy2.robjects.FloatVector(BKG_PRCP_QCd) #.................................. Vector of model background precip at observations
    r_lon_vec = rpy2.robjects.FloatVector(STAT_LON_QCd) #.................................. Vector of observation longitudes
    r_lat_vec = rpy2.robjects.FloatVector(STAT_LAT_QCd) #.................................. Vector of observation latitudes
    r_dataframe = rpy2.robjects.DataFrame({}) #............................................ R Dataframe initialized as null
    #
    d = {'lon': r_lon_vec, 'lat': r_lat_vec,'obs':r_obs_vec,'bkg':r_bkg_vec} #............. Dataframe library
    r_dataframe = rpy2.robjects.DataFrame(d) #............................................. R Dataframe of input data
    #
    r_gridlon_vec = rpy2.robjects.FloatVector(np.reshape(GRID_LONS,(nx*ny,1)).squeeze()) #. Vector of kriging grid longitudes
    r_gridlat_vec = rpy2.robjects.FloatVector(np.reshape(GRID_LATS,(nx*ny,1)).squeeze()) #. Vector of kriging grid latitudes
    r_gridbkg_vec = rpy2.robjects.FloatVector(np.reshape(GRID_BKG_APCP,(nx*ny,1)).squeeze()) #. Vector of kriging grid model background precip
    r_grid_dframe = rpy2.robjects.DataFrame({}) #.......................................... R Dataframe initialzied as null
    #
    d = {'lon': r_gridlon_vec, 'lat': r_gridlat_vec, 'bkg':r_gridbkg_vec} #................ Dataframe library
    r_grid_dframe = rpy2.robjects.DataFrame(d) #........................................... R Dataframe of output grid data
    #
    # Create an R function to convert R Dataframe to R Spatial Dataframe by defining
    # coordinates and assigning projection
    #
    rpy2.robjects.r('''

      project_var <- function(points) {
          coordinates(points) <- ~lon + lat
          proj4string(points) <- CRS("+proj=longlat")
          return(points)
       }
    ''')
    #
    # Assign R function to Python function and pass r_dataframe to create an R Spatial
    # Dataframe
    #
    project_var = rpy2.robjects.globalenv['project_var'] #................................ Python function calling R function
    r_s4 = project_var(r_dataframe) #..................................................... R Spatial Dataframe of input data
    #
    # Create an R function to compute empirical variogram
    #
    rpy2.robjects.r('''

      project_var <- function(s4_frame) {
          lzn.vgm <- variogram(obs~1,s4_frame,cressie=TRUE)
          return(lzn.vgm)
       }
    ''')
    #
    # Assign R function to Python function and pass r_s4 to create empirical variogram
    #
    project_var = rpy2.robjects.globalenv['project_var'] #................................ Python function callig R function
    r_vario = project_var(r_s4) #......................................................... R variogram
    #
    # Create R function to fit empirical variogram to spherical variogram model
    #
    rpy2.robjects.r('''

      project_var <- function(vario) {
          lzn.fit <- fit.variogram(vario, model=vgm("Sph"), fit.method=7)
          return(lzn.fit)
       }
    ''')
    #
    # Assign R function to Python function and pass r_vario to create fitted variogram
    #
    project_var = rpy2.robjects.globalenv['project_var'] #................................. Python function calling R function
    r_vario_fit = project_var(r_vario) #.................................................. R fitted variogram
    #
    # Create R function to convert R Dataframe to R Spatial Dataframe by defining
    # coordinates and assigning projection
    #
    rpy2.robjects.r('''

      project_var <- function(points) {
          coordinates(points) <- ~lon + lat
          proj4string(points) <- CRS("+proj=longlat")
          return(points)
       }
    ''')
    #
    # Assign R function to Python function and pass r_dataframe to create an R Spatial
    # Dataframe
    #
    project_var = rpy2.robjects.globalenv['project_var'] #................................. Python function calling R function
    r_grid_s4 = project_var(r_grid_dframe) #............................................... R Spatial Dataframe of output grid data
    #
    # Create R function to krige observations onto output grid
    #
    rpy2.robjects.r('''

      project_var <- function(s4_frame,s4_grid_frame,variofit) {
          lzn.kriged <- krige(obs ~ bkg, s4_frame, s4_grid_frame, model=variofit)
          return(lzn.kriged)
       }
    ''')
    #
    # Assign R function to Python function and pass Spatial Dataframes and fitted variogram
    # to create kriged data
    #
    project_var = rpy2.robjects.globalenv['project_var'] #................................. Python function calling R function
    r_krig = project_var(r_s4,r_grid_s4,r_vario_fit) #..................................... R Dataframe containing Kriged field and Kriging variance
    #
    r_krig_dframe=r_krig.slots['data'] #................................................... R Dataframe of kriging field and kriging variance
    #
    # Create R function to convert R Dataframe to R matrix
    #
    rpy2.robjects.r('''

      project_var <- function(dframe) {
          dmat = data.matrix(dframe)
          return(dmat)
       }
    ''')
    #
    # Assign R function to Python function and pass r_krig_dataframe to create r_krig_matrix
    #
    project_var = rpy2.robjects.globalenv['project_var'] #................................. Python function calling R function
    r_krig_matrix=project_var(r_krig_dframe) #............................................. R matrix containing kriging field and kriging data
    #
    # Convert r_krig_matrix to numpy matrix and extract fields
    #
    krig_data=np.asarray(r_krig_matrix) #.................................................. Numpy array containing 2 vectors: [0]: kriged field, [1]: kriging variance
    KRIG_FLD = krig_data[:,0] #............................................................ Kriged field (ny*nx,1)
    KRIG_VAR = krig_data[:,1] #............................................................ Kriging variance (ny*nx,1)
    #
    # At each grid-point, compute the 5% and 95% confidence bounds on the kriged field,
    # using the kriging variance inflated by INFL_FACT. This is accomplished by computing a
    # normal distribution at each grid point
    #
    # Compute normal distribution at each grid-point, with mean of KRIG_FLD and variance of
    # INFL_FACT * KRIG_VAR
    #
    rv=norm( #............................................................................. Normal distribution at each grid point
             loc=KRIG_FLD ,                        # Input Variable: Mean of normal distribution at point i
             scale=np.sqrt(INFL_FACT * KRIG_VAR)   # Input Variable: Standard deviation of normal distribution at point i
           )
    #
    # Compute 5% and 95% confidence bounds of KRIG_FLD at each point, based on normal
    # distribution
    #
    KRIG_FLD_05 = rv.ppf(0.05) #........................................................... 5% confidence bound on normal distribution at grid point i
    KRIG_FLD_95 = rv.ppf(0.95) #........................................................... 95% confidence bound on normal distribution at grid point i
    #
    # Reshape kriging field, confidence bounds, and kriging variance to (ny,nx) format
    #
    KRIG_FLD = np.reshape(KRIG_FLD,(ny,nx))
    KRIG_FLD_05 = np.reshape(KRIG_FLD_05,(ny,nx))
    KRIG_FLD_95 = np.reshape(KRIG_FLD_95,(ny,nx))
    KRIG_VAR = np.reshape(KRIG_VAR,(ny,nx))
    #
    # Compute analysis and bounds as BKG_APCP_LR + KRIG
    #
    HAN1_APCP = GRID_BKG_APCP + KRIG_FLD #................................................. Best-guess QPE analysis
    HAN05_APCP = GRID_BKG_APCP + KRIG_FLD_05 #............................................. 5% confidence bound on QPE analysis
    HAN95_APCP = GRID_BKG_APCP + KRIG_FLD_95 #............................................. 95% confidence bound on QPE analysis
    #
    # Reset analysis values below zero to zero
    #
    HAN1_APCP[HAN1_APCP<0.] = 0.
    HAN05_APCP[HAN05_APCP<0.] = 0.
    HAN95_APCP[HAN95_APCP<0.] = 0.
    #
    # Pull variogram information to numpy arrays
    #
    #
    # Create an R function to compute fitted variogram values
    #
    rpy2.robjects.r('''
      
      project_var <- function(vario,vario_fit) {
          vario_fit.line <- variogramLine(vario_fit,maxdist=max(vario$dist),dist_vector=vario$dist)
          return(vario_fit.line)
       }
    ''')
    #
    # Assign R function to Python function and pass r_vario, r_vario_fit to create fitted 
    # variogram values
    #
    project_var = rpy2.robjects.globalenv['project_var'] #................................. Python function callig R function
    r_vario_line = project_var(r_vario,r_vario_fit) #...................................... R fitted-variogram values
    #
    # Define empirical (EMP_) and fitted (FIT_) variogram values
    #
    EMP_VARIO_PAIRS = np.asarray(r_vario[0]).squeeze() #................................... Empirical variogram number of pairs per distance bin
    EMP_VARIO_DISTS = np.asarray(r_vario[1]).squeeze() #................................... Empirical variogram distance bin (centers, km)
    EMP_VARIO_GAMMA = np.asarray(r_vario[2]).squeeze() #................................... Empirical variogram semivariance per distance bin
    #
    FIT_VARIO_GAMMA = np.asarray(r_vario_line[1]).squeeze() #.............................. Fitted variogram semivariance per distance bin
    #
    # Fitted nugget (semi-)variance is the first value of the second column of r_vario_fit
    # i.e., the partial sill of the Nugget model
    #
    FIT_VARIO_NUGGE = r_vario_fit[1][0] #.................................................. Fitted variogram nugget (semi-)variance
    #
    # Return
    #
    return KRIG_FLD , KRIG_FLD_05 , KRIG_FLD_95 , KRIG_VAR , HAN1_APCP , HAN05_APCP , \
           HAN95_APCP , EMP_VARIO_PAIRS , EMP_VARIO_DISTS , EMP_VARIO_GAMMA , \
           FIT_VARIO_GAMMA , FIT_VARIO_NUGGE
#
###########################################################################################
#
