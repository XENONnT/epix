import numpy as np
import pandas as pd
import numba 
import awkward as ak
from .common import reshape_awkward



def event_lineage(interactions, distance_threshold):
    """
    Function which iterates over all events and builds the track lineage

    Args:
        interactions (awkward.Array): Array containing at least the following
            fields: x,y,z, type, parenttype, trackid, types
        distance_threshold (float): max allowed distance between parent and daughter interaction
            to stay in the same lineage

    Returns:
        awkward.array: Adds to interaction a lineage_ids record.
    """
    
    all_lineages = []

    for inter in interactions:
        trackid = np.array(inter["trackid"])
        parentid = np.array(inter["parentid"])
        types = np.array(inter["type"])
        x = np.array(inter["x"])
        y = np.array(inter["y"])
        z = np.array(inter["z"])

        lineage = build_lineage(trackid, types, parentid, x, y, z, distance_threshold)
        all_lineages.append(lineage)

    all_lineages = np.concatenate(all_lineages)
    interactions['lineage_ids'] = reshape_awkward(all_lineages, ak.num(interactions['x']))

    return interactions

def build_lineage(trackid, types, parentid, x, y, z, distance_threshold):
    """
    Function which builds the lineage within an event.

    Args:
        trackid (np.array): array containing the trackid comming from Geant4 
        types (np.array): array containing the particle types comming from Geant4
        parentid (np.array): array containing the particles parent id comming from Geant4
        x (np.array): x position of the interaction
        y (np.array): y position of the interaction
        z (np.array): z position of the interaction
        distance_threshold (float): max allowed distance between parent and daughter interaction
            to stay in the same lineage

    Returns:
        np.array: A numpy array containing the lineage of a single event. 
    """
    n_interactions = len(trackid)
    
    #set up the new legacy array
    lineage = np.zeros(n_interactions, dtype = np.int)
    
    #iterate over all interactions in the event:
    for idx in range(n_interactions):
        
        #particle_type = types[idx]
        
        #1 check if particle has a parent particle 
        particle_parent_type_array = types[trackid==parentid[idx]]
        if particle_parent_type_array.size > 0:
            particle_parent_type = particle_parent_type_array[0]
        else:
            #This particle has no parent
            #Need to start a new lineage
            lineage[idx] += np.max(lineage)+1
            
            own_lineage = lineage[trackid==trackid[idx]]
            if own_lineage.size > 0:
                lineage[idx] = own_lineage[0]
            
            continue
        
        parent_lineage = lineage[trackid==parentid[idx]][0]
        
        #distance between parent and daughter
        parent_position = np.array([x[trackid==parentid[idx]][-1],
                                    y[trackid==parentid[idx]][-1],
                                    z[trackid==parentid[idx]][-1]
                                   ])
        
        particle_position = np.array([x[idx],
                                      y[idx],
                                      z[idx],
                                     ])

        distance = np.sqrt(np.sum((parent_position-particle_position)**2, axis=0))
        
        #2 investigate lineage of parent particle
        inherit_lineage = check_inheritance(particle_parent_type, distance, distance_threshold)
        
        if inherit_lineage == False:
            #start a new legacy 
            lineage[idx] += np.max(lineage)+1
        #3 set lineage to parent legacy
        else:
            lineage[idx] = parent_lineage
            
        #fix lineage if particle has multiple interactions..
        # lets make it more clear in the future
        own_lineage = lineage[trackid==trackid[idx]]
        if own_lineage.size > 0:
            lineage[idx] = own_lineage[0]
        
    return lineage

def check_inheritance(particle_parent_type, distance, distance_threshold):
    """
    To do in this function:
    -Check if distance between parent and daughter is > a threshold - if so- new legacy
    -Check if particle passes legacy to daughters 
    
    Return True if legacy is passed to daughters
    Return False if it is not passed and a new legacy will be started
    """
    
    #Lets say for now, that only electrons and gammas pass their legacy 
    #Need to check again what nest has done
    if particle_parent_type not in ["e-", "gamma"]:
        return False
    
    if distance>distance_threshold:
        return False
    
    else:
        return True