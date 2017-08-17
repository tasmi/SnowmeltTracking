"""
Detect_Melt.py

Functions for determining the start (proxied by the maximum XPGR) and end of the snowmelt season from passive microwave data
Author: Taylor Smith (tasmith@uni-potsdam.de)
Date, v1.1: 17.8.2017

"""

import numpy as np
import pandas as pd
from detect_peaks import detect_peaks #This function can be found at https://github.com/demotu/BMC
import traceback, datetime

def daterangelist(start_date, end_date, skip):
    """ 
    Helper Function to return a list of dates
    given a start_date, end_date, skip value in days
    Returns: A Python List of datetime64 objects    
    """
    l = np.array([])
    for n in range(0, int((end_date - start_date).days), skip):
        l = np.append(l, (start_date + datetime.timedelta(n)))
    return l.astype('datetime64')
    
def Melt_Dates2(XPGR, SWE, Tb37V, END):
    """Classify the start (MXPGR) and end of melt based upon XPGR, SWE, 37V PM data, and the long-term average end date.
    For classifying the long-term average end date, use 'NULL' in place of END.
    
    Inputs: XPGR, SWE, Tb37V, END (or NULL for long-term mean dates)
    Returns: Onset, End Date, quality flag (1 = good, 0 = unconfirmed melt onset), subpeak (either date of secondary peak or zero if no secondary peak)
    """
    def start(XPGR):
        """Find the maximum XPGR / widest peak from a given XPGR time series"""
        peaks = detect_peaks(XPGR.values) #Identify peaks in XPGR data
        aoi_sum = -100 #Set initial condition
        w = 0 #Set a flag to check to make sure a peak is found
        for p in peaks:
            peak = XPGR.index[p]
            aoi = XPGR[np.logical_and(XPGR.index < peak + pd.Timedelta('2 days'), XPGR.index > peak - pd.Timedelta('2 days'))] #Select values within 5 day window
            if np.nanmean(aoi.values) > aoi_sum: 
                aoi_sum = np.nanmean(aoi.values) #Replace the initial condition if the peak is higher/thicker
                current_max = peak
                w = 1 #Set the flag to say a peak has been found
        if w == 0: #If current max is never assigned, return the yearly max instead
            current_max = XPGR.idxmax()
            
        #Check if the peak is the only strong peak in a year
        maxval = np.nanmax(XPGR.values[np.logical_and(XPGR.index < current_max + pd.Timedelta('5 days'), XPGR.index > current_max - pd.Timedelta('5 days'))]) #Max val around the chosen peak
        subpeaks_possible = XPGR[np.logical_or(XPGR.index > current_max + pd.Timedelta('21 days'), XPGR.index < current_max - pd.Timedelta('21 days'))] #Only look for peaks more than 3 weeks away from the chosen peak
        offset = 0.95 * maxval #Set the allowable offset to 5% of the max
        subpeaks = subpeaks_possible[subpeaks_possible.values > offset] #Find possible secondary peaks
        
        flag = 1 #Set the default flag at good
        if subpeaks.shape[0] > 2:
            flag = 0 #If there are several other days above the max, call the MXPGR unconfirmed
        try:
            subpeak = subpeaks.idxmax() #Try to return the highest subpeak, otherwise return a zero
        except:
            subpeak = 0
                                
        return current_max, flag, subpeak
    
    def end(Tb37V, SWE):
        """Find the end date of melt from given 37V and SWE time series"""
        peaks = detect_peaks(Tb37V.values, mph=0) #Find all positive peaks in 37V data
        aoi_sum = 0 #Set initial condition
        for p in peaks:
            peak = Tb37V.index[p]
            aoi = Tb37V[np.logical_and(Tb37V.index < peak + pd.Timedelta('3 days'), Tb37V.index > peak - pd.Timedelta('3 days'))] #Select values within 3 days
            if np.nanmean(aoi.values) > aoi_sum: 
                aoi_sum = np.nanmean(aoi.values) #Replace initial condition if the peak is higher/thicker
                current_max = peak
        RollSWE = SWE.copy() #Create a copy of the SWE data
        minswe = np.nanmin(SWE.values) #Find yearly minimum SWE
        if np.isnan(np.nanmin(SWE.values)):
            minswe = 0
        RollSWE[SWE.values <= minswe + 2] = 1 #Classify the SWE series into 1s and 0s based on distance from min swe
        RollSWE[SWE.values > minswe + 2] = 0
        rm = pd.rolling_sum(RollSWE, 5) #Calculate a rolling sum on a 5 day window
        snow_clear = rm[rm.values >= 4].index.min() - pd.Timedelta('4 days') #Declear snow clearance if 4/5 days are 'snow clear'
        try:
            return pd.Series((snow_clear, current_max)).min() #Choose the minimum between the SWE clearance and the Tb37 max dates
        except:
            try:
                return pd.Series(snow_clear, Tb37V.idxmax()).min() #If no data was returned from the Tb37 peak find, use the maximum yearly value instead
            except:
                return snow_clear #If this also fails, use only the snow clearance date to determine date of snow melt off
          
    if END != 'NULL': #Main classifier
        endmelt = end(Tb37V, SWE)
        if np.abs((END - endmelt).days) > 60: #If there is high deviation from long-term average end dates, reclassify using a threshold
            m, flag, subpeak = start(XPGR)
            endmelt = end(Tb37V[Tb37V.index > m], SWE[SWE.index > m])
            try:
                m, flag, subpeak = start(XPGR[XPGR.index < endmelt])
            except:
                pass
        else:
            try:
                m, flag, subpeak = start(XPGR[XPGR.index < endmelt])
            except:
                m, flag, subpeak = start(XPGR)
    elif END == 'NULL': #Initial use, when long-term average time series is used instead of yearly data
        m, flag, subpeak = start(XPGR) #First guess of MXPGR
        endmelt = end(Tb37V, SWE) #First guess of melt end
        if m > endmelt: #If the date of MXPGR is later than the date of melt end, reclassify the melt end date using only the data following the onset of melt
            endmelt = end(Tb37V[Tb37V.index > m], SWE[SWE.index > m])
        else:
            m, flag, subpeak = start(XPGR[XPGR.index < endmelt]) #If the date of melt onset is before the date of melt end, reclassify the start of the melt season only with those data before the end of melt

    return m, endmelt, flag, subpeak

def Average_Year(ser):
    """Generate an average, long-term year from a given pandas time series"""
    Average_Year = ser.groupby([ser.index.month, ser.index.day]).mean() #Get average value at each day of year over whole series length
    try:
        Average_Year.index = daterangelist(datetime.datetime(2000,1,1), datetime.datetime(2001,1,1),1) #Fit these days onto a 366 day series (to account for leap years)
    except:
        minus = 366 - Average_Year.index.shape[0]
        Average_Year.index = daterangelist(datetime.datetime(2000,1,1), datetime.datetime(2001,1,1) - pd.Timedelta(str(minus) + ' days'),1) #If your series doesn't contain each day of the year, shrink the series
    av = Average_Year.index.copy()
    idx = np.where(av > pd.Timestamp('2000-09-30'))
    idx2 = np.where(av <= pd.Timestamp('2000-09-30'))
    new = av[idx].map(lambda t: t.replace(year=1999, month=t.month, day=t.day))
    new2 = av[idx2].map(lambda t: t.replace(year=2000, month=t.month, day=t.day))
    Av = pd.Series(np.concatenate((Average_Year.values[idx2],Average_Year.values[idx])), index=np.concatenate((new2,new)))
    Av = Av.sort_index() #Reshape the matrix so that it is as a 'water year', running from Oct 1 to Sept 30
    
    return Av
    
def WaterYear(series):
    """ 
    Takes a pandas series and splits it into water-year segments.
    Returns a list of pandas series
    """
    chunked = []
    for yr in np.unique(series.index.year):
        if yr == np.min(np.unique(series.index.year)):
            S = series[np.logical_and(series.index.year == np.min(np.unique(series.index.year)), series.index.month < 10)]
        else:
            S = pd.concat([series[np.logical_and(series.index.year == (yr - 1), series.index.month > 9)], series[np.logical_and(series.index.year == yr, series.index.month < 10)]])
        chunked.append(S)
    return chunked
        
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Takes input of series, window_size, order, deriv, and rate
    DEFAULT: savitzky_golay(ser.values, 21, 1, deriv=0, rate=1)
    window_size must be ODD
    """
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
    
#Example Usage:
def MeltDates(XPGR, SWE, Tb37):
    """ Example function to take XPGR, SWE, and Tb37V data as pandas timeseries to track the
    onset (MXPGR) and end of melt on a year-by-year basis.
    Inputs are pandas Series objects.
    """
    Dict = {}
    #Compute Average Year
    Av = Average_Year(XPGR) 
    AveSWE = Average_Year(SWE)
    Ave37 = Average_Year(Tb37)
    avem, aveend, flag, subpeak = Melt_Dates(Av, AveSWE, Ave37, 'NULL')
    avem = pd.Timestamp(avem, tz='UTC') - pd.Timestamp('1999-10-01', tz='UTC') #Change the average date to a 'distance from Oct1 metric'
    aveend = pd.Timestamp(aveend, tz='UTC') - pd.Timestamp('1999-10-01', tz='UTC')
    
    longavstart = avem.days
    longavend = aveend.days

    chunk_XPGR = WaterYear(XPGR)
    chunk_37 = WaterYear(Tb37)
    chunk_swe = WaterYear(SWE)
    try:
        num = len(chunk_XPGR)
    except:
        num = 1
    for ii in range(0,num):
        #For each year of data... 
        XPGR = chunk_XPGR[ii]
        Tb37V = chunk_37[ii]
        SWE = chunk_swe[ii]
        if not XPGR.index.year.max() in baddict[inst]:
            year = XPGR.index.year.max()
            #Generate expected start dates from long-term averages...
            START = avem + XPGR.index.min()
            END = aveend + XPGR.index.min()
            
            m, endmelt, flag, subpeak = Melt_Dates(XPGR, SWE, Tb37V, END)
            try:
                if (endmelt - m).days > 0 and flag == 1:
                    oct1 = pd.Timestamp('%s-10-01' % XPGR.index.year.min()) #Oct 1, start of water year
                    distfromstart = (m - oct1).days
                    distfromend = (endmelt - oct1).days
                    period = (endmelt - m).days
                    startdeviation = (m - START).days
                    enddeviation = (endmelt - END).days
                    #Clean unreasonable values...
                    if period > 366:
                        period = np.nan
                    if distfromstart > 366:
                        distfromstart = np.nan
                    if distfromend > 366:
                        distfromend = np.nan
                    if period < 0:
                        period = np.nan
                    if distfromstart < 0:
                        distfromstart = np.nan
                    if distfromend < 0:
                        distfromend = np.nan
                else:
                    distfromstart, distfromend, period = (np.nan,)*3
            except:
                distfromstart, distfromend, period = (np.nan,)*3
                traceback.print_exc()
            storedata = [period, distfromstart, distfromend]
            Dict[str(year)] = storedata #Save the data to a by-year dictionary
            del XPGR, Tb37V, SWE
    DictOut = {}
    for yr in range(1988,2017):
        periods, starts, ends, flgs = [], [], [], [] #Now have the start, end, per and quality flag for each inst/year combo
        for i in insts:
            try:
                data = Dict[i + str(yr)]
                periods.append(data[0])
                starts.append(data[1])
                ends.append(data[2])
                flgs.append(data[3])
            except:
                pass
        #Range check
        datelist = [periods, starts, ends]
        indlist = [longavend - longavstart, longavstart, longavend] #Longterm average period, MXPGR, end
        outlist = []
        for i in [0,1,2]:
            try:
                l = np.array(datelist[i]).astype(float) #Grab all instrument values for a given year
                if i != 2:
                    if not np.nansum(flgs) == np.array(flgs).shape[0]: #If not every onset value is well-defined, throw them all out
                        l = (np.nan,)*np.array(flgs).shape[0]
                if len(l) > 1: #If there is more than one...
                    try:
                        if np.nanmax(l) - np.nanmin(l) > 14: #If the spread between high quality dates is high...
                            if l.shape[0] > 2:
                                   l = np.nanmedian(l)
                            else:
                                if i == 2: #For the end dates use a different metric as they are better constrained
                                    ind = indlist[i] #Pull the long-term average
                                    finder = np.abs(np.array(l) - ind) #Get the absolute distance from the long-term average
                                    l = l[np.where(finder == np.nanmin(finder))[0]][0] #Choose the date that is closest to the long-term average
                                else:
                                    l = np.nan
                        else:
                            l = l[0] #Choose the earlier date when onset dates and periods are close together
                    except:
                        l = l[0] 
                else: #If there is only one, return the value as a float
                    l = float(l[0]) 
            except:
                l = np.nan
            outlist.append(l)
        DictOut[str(yr)] = outlist
        
    return DictOut #Return the data in the format Dictionary[year] = [period, MXPGR, end]
    
    
