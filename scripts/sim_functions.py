import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import pandas as pd


def make_gene_body_index(gene_body_length, terminator_length):
    '''
        This function makes the gene body index. It returns a list of 1 and 2, which are indices for 
        the gene body vs. terminator, respectively.
    '''
    # gene_body_length = 10
    # terminator_length = 20
    gene_body_coords = ['gene_body' for i in range(gene_body_length)]
    middle_term_rev = ['term_rev' for i in range(terminator_length)]
    middle_term_for = ['term_for' for i in range(terminator_length)]
    gene_coordinates = gene_body_coords + middle_term_rev + middle_term_for + gene_body_coords
    
    return gene_coordinates

# make_gene_body_index(gene_body_length=10, terminator_length=10)

def make_par_regime(stay_prob, forward_prob, release_prob, eps_val = 1e-10, eps_val2 = 1e-5):
    '''
        This function is for formatting probabilities of movements (transition matrix) for RNAP.
    '''
    
    ## check that probs sum to 1
    total_sum = round(sum([stay_prob, forward_prob, release_prob]), 6)

    if total_sum != 1:
        raise ValueError("Probabilities do not sum to 1, check those out you goose.")
    
    ## format probabilities
    out_dict = {'stay_pr': stay_prob,
               'forward_pr': forward_prob,
               'release_pr': release_prob}
    
    return out_dict

def make_look_up_for_gene_body(stay_prob_gb, forward_prob_gb, release_prob_gb,
                               stay_prob_term, forward_prob_term, release_prob_term,
                               stay_prob_ogb, forward_prob_ogb, release_prob_ogb):
    '''
    This is for making a lookup dictionary
    '''
    
    ## Make parameter dictionary -- gene body
    gb_pars = make_par_regime(stay_prob = stay_prob_gb, 
                        forward_prob = forward_prob_gb, 
                        release_prob = release_prob_gb)
    
    ## Make parameter dictionary -- terminator
    term_pars = make_par_regime(stay_prob = stay_prob_term, 
                        forward_prob = forward_prob_term, 
                        release_prob = release_prob_term)
    
    ## Make parameter dictionary -- opposing gene body
    ogb_pars = make_par_regime(stay_prob = stay_prob_ogb, 
                        forward_prob = forward_prob_ogb, 
                        release_prob = release_prob_ogb)
    
    return {'gene_body':gb_pars, 
            'term_rev':term_pars,
            'term_for':term_pars,
            'ogene_body':ogb_pars}

class RNAP:
    ## current position
    position = 0
    ## starting position
    start_position = 0
    ## terminated position
    finish_position = 0
    ## current parameter regime
    par_regime = 'gene_body'
    ## direction
    direction = 'forward'
    ## indicator if terminated because of bumping
    bumped_when_term = 0
    # number of time steps where stalled
    stalled_from_head_out = 0
    time_step_added = 0
    ## through a terminator
    through_a_term = False

def increment_forward_individual(ind_rnap, gene_map, par_list, ind_rnap_list, bump_distance, 
                                 last_for_term_position, last_rev_term_position, ## these are calculated from the gene_map
                                 init_val = False):
    '''
        loops through an individual and then adjusts positions based on transition probabilities
        also needs the ind_rnap_list to accomodate when a trailing rnap bumps it off, or stalls
        because of head on collision
    '''
    
    ## adjust the parameter regime to be aligned
    set_par_regime = gene_map[ind_rnap.position] #(gene_map is a list of par regime names)
    ind_rnap.par_regime = set_par_regime
    
    ## adjust to be considered gene body pars depending on direction. this is necessary so that the
    ## terminator comes *after* the RNAP goes over it.
    if ind_rnap.par_regime == 'term_for' and ind_rnap.direction == 'reverse':
        ind_rnap.par_regime = 'gene_body'
    
    if ind_rnap.par_regime == 'term_rev' and ind_rnap.direction == 'forward':
        ind_rnap.par_regime = 'gene_body'
        
    ## add statements about a parameter regime test if in gene body and past terminator
    if ind_rnap.direction == 'forward' and ind_rnap.position > last_for_term_position:
        ind_rnap.par_regime = 'ogene_body'
    if ind_rnap.direction == 'reverse' and ind_rnap.position < last_rev_term_position:
        ind_rnap.par_regime = 'ogene_body'
    
    ## get an action outcome based on the parameter regime probabilities
    action_outcome = np.random.choice(list(par_list[ind_rnap.par_regime].keys()), 
                                      1,
                                      p = list(par_list[ind_rnap.par_regime].values())).tolist()
    
    ## If this is not an initialization
    if not init_val:
        
        ## set the action outcome to stay if there is a head on RNAP going opposite direction.
        df_of_head_on = filter_df_based_on_distance_head_on(position_list_df = make_df_positions(ind_rnap_list), 
                                             target_position = ind_rnap.position, 
                                             target_direction = ind_rnap.direction, 
                                             distance_to_bump = bump_distance)

        ## how many head on bumpers are there?
        num_rows_bumpers = df_of_head_on.shape[0]

        ## if there is a head on bumping up, then pause
        if num_rows_bumpers > 0:
            action_outcome[0] = 'stay_pr'
            ind_rnap.stalled_from_head_out += 1
        
        # set the action outcome to release if there is a same-direction RNAP within x-nucleotides.
        df_of_trailers = filter_df_based_on_distance_trailing(position_list_df = make_df_positions(ind_rnap_list), 
                                             target_position = ind_rnap.position, 
                                             target_direction = ind_rnap.direction, 
                                             distance_to_bump = bump_distance)

        ## how many trailers are there?
        num_rows = df_of_trailers.shape[0]

        ## if there is a trailer in the bumping distance, then release
        if num_rows > 1:
            action_outcome[0] = 'release_pr'
            ind_rnap.bumped_when_term += 1
    
    ## change position based on this action outcome (which was either prob. altered, or set because of other RNAPs)
    if action_outcome[0] == 'stay_pr':
        ind_rnap.position = ind_rnap.position
        
    ## if move forward, that depends on the direction
    if action_outcome[0] == 'forward_pr':
        if ind_rnap.direction == 'forward':
            ind_rnap.position += 1
        if ind_rnap.direction == 'reverse':
            ind_rnap.position -= 1
            
    ## if released
    if action_outcome[0] == 'release_pr':
        ind_rnap.finish_position = ind_rnap.position
        
    ## if the position is out of bounds
    if ind_rnap.position == len(gene_map) and ind_rnap.direction == 'forward':
        ind_rnap.finish_position = ind_rnap.position
    if ind_rnap.position == 0 and ind_rnap.direction == 'reverse':
        ## using -1 here, instead of 0, because I want to be compatible with the increment time function.
        ind_rnap.finish_position = -1
        
    return(ind_rnap)

def show_all_attributes(rnap_obj):
    '''
    handy function for looking at individual rnap objects
    '''
    print(rnap_obj.position)
    print(rnap_obj.start_position)
    print(rnap_obj.finish_position)
    print(rnap_obj.par_regime)
    print(rnap_obj.direction)
    print(rnap_obj.bumped_when_term)

def new_end_rnap(gene_map_input):
    '''
        this function just makes an rnap object but with end characteristics
    '''
    # Make a new RNAP object that starts at the end of the gene body and goes reverse
    end_RNAP = RNAP()
            
    # Add the minus 1 because of zero indexing
    end_RNAP.position = len(gene_map_input) - 1
    end_RNAP.direction = 'reverse'
    end_RNAP.start_position = len(gene_map_input) - 1
    
    return end_RNAP

def increment_time(ind_rnap_list, ind_rnap_list_off, gene_map, par_list, n_steps, 
                   on_rate_per_time_step_rev, on_rate_per_time_step_for, 
                   bump_distance, unidirectional = False):
    '''
        run simulation forward in time, first looping through different time steps
        and then looping through individual RNAP molecules
        
    '''
    
    ## calculate the terminator positions for a given gene map
    last_for_term_position_gm = max(i for i, term in enumerate(gene_map) if term == 'term_for')
    last_rev_term_position_gm = min(i for i, term in enumerate(gene_map) if term == 'term_rev')
    
    for i in range(n_steps):
        
        ## this is to test why some RNAPs are being bumped
#         if check_if_any_rnap_bumped(position_list = ind_rnap_list) == 'just_bumped':
#             break
        
        rnap_occ = check_if_new_rnap_occupied(position_list = ind_rnap_list, 
                                   distance_to_bump = bump_distance, 
                                   gene_map_length = len(gene_map))
        
        ## probabilistically adding an RNAP to the forward gene
        r_beginning = bernoulli.rvs(1/on_rate_per_time_step_for, size=1)
        
        ## if there is an RNAP that attached *and* there is not a very recently added RNAP, then go ahead! Yay!
        if r_beginning[0] == 1 and not rnap_occ[0]:
            ## Add RNAP to the beginning of the list
            new_beginning_rnap = RNAP()
            ## Include what time it was added
            new_beginning_rnap.time_step_added = i
            ## add to the list
            ind_rnap_list.insert(0, new_beginning_rnap)
            
        ## probabilistically adding an RNAP to the reverse gene
        r_ending = bernoulli.rvs(1/on_rate_per_time_step_rev, size=1)
        ## if it's not a single direction simulation
        if not unidirectional:
            if r_ending[0] == 1 and not rnap_occ[1]:
                # Add RNAP to the end of the list
                new_end_rnap_i = new_end_rnap(gene_map_input = gene_map)
                ## Include what time it was added
                new_end_rnap_i.time_step_added = i
                ## add to the list
                ind_rnap_list.append(new_end_rnap_i)
            
        ## starting at the end of the list and going backwards    
        for j in range(len(ind_rnap_list) - 1, -1, -1):
            
            ## increment individual rnap molecules
            ind_rnap_list[j] = increment_forward_individual(ind_rnap_list[j], gene_map, par_list, 
                                                            ind_rnap_list = ind_rnap_list,
                                                            bump_distance = bump_distance, 
                                                            last_for_term_position = last_for_term_position_gm, 
                                                            last_rev_term_position = last_rev_term_position_gm)
            
            ## if the release probability was chosen, then finish_position is incremented to it's no longer zero.
            ## this RNAP is off the gene, see you later!!
            if ind_rnap_list[j].finish_position != 0:
            
                ## add to the inactive one
                ind_rnap_list_off.append(ind_rnap_list[j])
                
                ## remove that from the active individual RNAP list
                ind_rnap_list.pop(j)
                
        ## now update position list %% hmm not sure what this does anymore
        latest_position_list = get_rnap_position(rnap_individual_list = ind_rnap_list)

    return [ind_rnap_list, ind_rnap_list_off]

def make_df_positions(position_list):
    '''
    position list dataframe (in pandas) from vector of rnap objects. 
    This is used for determining trailing or head on RNAPs
    '''
    
    direction_list = []
    position_on_gene_body_list = []
    
    for i in range(len(position_list)):
        direction_list.append(position_list[i].direction)
        position_on_gene_body_list.append(position_list[i].position)
        
    out_df = pd.DataFrame({'direction': direction_list,
                           'position': position_on_gene_body_list})

    return out_df

def check_if_any_rnap_bumped(position_list):
    '''
    takes in list of rnaps active and scans to see if any have been bumped.
    this function is for testing.
    '''
    state_out = 'no_bumps'
    
    for i in position_list:
        if i.bumped_when_term > 0:
            state_out = 'just_bumped'
            break
            
    return state_out
    

def check_if_new_rnap_occupied(position_list, distance_to_bump, gene_map_length, buffer_distance = 5):
    '''
    output True or False if there are no RNAPs within bumping distance of the initiation site
    output has two True and False entries, the first is for the forward gene, the second
    is for the reverse gene.
    '''
    position_list_df = make_df_positions(position_list)
    
    output_df_for = position_list_df[(position_list_df['position'] < distance_to_bump + buffer_distance)]
    output_df_rev = position_list_df[(position_list_df['position'] > gene_map_length - distance_to_bump - buffer_distance)]
    
    out_val_for = False
    out_val_rev = False
    
    if output_df_for.shape[0] > 0:
        out_val_for = True
    if output_df_rev.shape[0] > 0:
        out_val_rev = True
        
    return [out_val_for, out_val_rev]

def filter_df_based_on_distance_trailing(position_list_df, target_position, 
                                         target_direction, distance_to_bump):
    '''
    convert to a dataframe of the nearby, trailing
    '''
    ## first filter by direction -- we only want the trailing polymerase
    pos_list_dir_filter = position_list_df[(position_list_df['direction'] == target_direction)]
    
    ## what you filter / or consider trailing depends on the direction
    if target_direction == 'forward':
        ## if it's going forward, we only take positions behind, hence the minus sign
        filtered_df = pos_list_dir_filter[(pos_list_dir_filter['position'] >= target_position - distance_to_bump) & 
                                          (pos_list_dir_filter['position'] <= target_position)]
        ## if it's going reverse, we only take positions above, hend the plus sign
    if target_direction == 'reverse':
        filtered_df = pos_list_dir_filter[(pos_list_dir_filter['position'] <= target_position + distance_to_bump) & 
                                          (pos_list_dir_filter['position'] >= target_position)]
        
    return filtered_df

def filter_df_based_on_distance_head_on(position_list_df, target_position, 
                                         target_direction, distance_to_bump):
    '''
    convert to a dataframe of the nearby, trailing
    '''
    if target_direction == 'forward':
        target_direction_opp = 'reverse'
    if target_direction == 'reverse':
        target_direction_opp = 'forward'
    
    ## filter the dataframe for only opposing direction RNAPs
    pos_list_dir_filter = position_list_df[(position_list_df['direction'] == target_direction_opp)]
    
    if target_direction == 'forward':
        ## if it's going forward, take the oncoming RNAPs that have a higher position value
        filtered_df = pos_list_dir_filter[(pos_list_dir_filter['position'] <= target_position + distance_to_bump) & 
                                          (pos_list_dir_filter['position'] >= target_position)]
        ## if it's going reverse, take the oncoming RNAPs: those that have a lower position value
    if target_direction == 'reverse':
        filtered_df = pos_list_dir_filter[(pos_list_dir_filter['position'] >= target_position - distance_to_bump) & 
                                          (pos_list_dir_filter['position'] <= target_position)]
        
    return filtered_df

def get_rnap_position(rnap_individual_list):
    '''
        This function takes in a list of rnap individuals whose positions are adjusted through time.
    '''
    ## make a position list
    rnap_position_list = []
    for i in rnap_individual_list:
        ## append the positions
        rnap_position_list.append(i.position)
    
def plot_hist_of_stopping_points(list_of_off_rnaps):
    '''
        loops through the list of off rnaps to map the termination position
    '''
    off_position_vals = []
    for i in list_of_off_rnaps[1]:
        off_position_vals.append(i.finish_position)
        
    on_position_vals = []
    for i in list_of_off_rnaps[1]:
        on_position_vals.append(i.start_position)
        
    # Create histogram
    plt.hist(off_position_vals + on_position_vals, 
             edgecolor='black',
            bins = 100)

    # Add titles and labels
    plt.xlabel('Genomic Position')
    plt.ylabel('Count')
    plt.title('Full length transcript start and finish points')

    # Show the plot
    plt.show()
    
def plot_hist_of_stopping_points_bumped(list_of_off_rnaps):
    '''
        loops through the list of off rnaps to map the termination position
    '''
    off_position_vals = []
    for i in list_of_off_rnaps[1]:
        if i.bumped_when_term > 0:
            off_position_vals.append(i.finish_position)
        
    on_position_vals = []
    for i in list_of_off_rnaps[1]:
        if i.bumped_when_term > 0:
            on_position_vals.append(i.start_position)
        
    # Create histogram
    plt.hist(off_position_vals + on_position_vals, 
             edgecolor='black',
            bins = 100)

    # Add titles and labels
    plt.xlabel('Genomic Position')
    plt.ylabel('Count')
    plt.title('Full length transcript start and finish points -- bumped RNAPs only')

    # Show the plot
    plt.show()
    
    
def plot_hist_of_active_rnaps(list_of_off_rnaps):
    '''
    actively transcribing rnaps
    '''
    ## now do the same thing but with transcribing RNAPs
    current_position_vals = []
    for i in list_of_off_rnaps[0]:
        current_position_vals.append(i.position)
        
    on_position_vals = []
    for i in list_of_off_rnaps[0]:
        on_position_vals.append(i.start_position)
        
    # Create histogram
    plt.hist(current_position_vals + on_position_vals, 
             edgecolor='black',
            bins = 100)

    # Add titles and labels
    plt.xlabel('Genomic Position')
    plt.ylabel('Count')
    plt.title('Transcribing RNAPs')

    # Show the plot
    plt.show()

def calculate_bumped_rnaps(list_of_off_rnaps):
    '''
    function to calculate the proportion of rnap release events from bumping
    '''
    bump_yes = 0
    for i in range(len(list_of_off_rnaps[1])):
        bump_yes += list_of_off_rnaps[1][i].bumped_when_term
    if len(list_of_off_rnaps[1]) > 0:
        proportion_bumped = bump_yes/len(list_of_off_rnaps[1])
    else:
        proportion_bumped = 'none_came_off'
    
    return proportion_bumped

def calculate_pausing_time(list_of_off_rnaps):
    '''
    function to calculate the proportion of rnap release events from bumping
    '''
    stalling_time = 0
    for i in range(len(list_of_off_rnaps[1])):
        stalling_time += list_of_off_rnaps[1][i].stalled_from_head_out
    
    return stalling_time

def get_sim_summary(list_of_off_rnaps):
    '''
    make plots and get summary statistics
    '''
    print('proportion of released rnaps that were bumped:', calculate_bumped_rnaps(list_of_off_rnaps))
    print('time spent pausing from oncoming:', calculate_pausing_time(list_of_off_rnaps))
    plot_hist_of_stopping_points(list_of_off_rnaps)
    plot_hist_of_stopping_points_bumped(list_of_off_rnaps)
    plot_hist_of_active_rnaps(list_of_off_rnaps)

def get_sim_summary_df(list_of_off_rnaps):
    '''
    Make a dataframe that summarizes terminated transcript boundaries
    '''
    
    direction_list = []
    finish_position_list = []
    start_position_list = []
    bumped_when_term_list = []
    
    for i in list_of_off_rnaps[1]:
        direction_list.append(i.direction)
        finish_position_list.append(i.finish_position)
        start_position_list.append(i.start_position)
        bumped_when_term_list.append(i.bumped_when_term)

        
    out_df = pd.DataFrame({'direction': direction_list,
                           'start_position': start_position_list,
                           'finish_position': finish_position_list,
                           'bumped': bumped_when_term_list})

    return out_df

def parameter_runs(gene_map, par_list, 
                   on_rate_per_time_step_rev_list = [120], on_rate_per_time_step_for_list = [120],
                   n_steps = 2000, bump_distance = 30, unidirectional=False):
    '''
    do a parameter scan
    '''
    list_of_pd_df = []
    
    for i in on_rate_per_time_step_rev_list:
        for j in on_rate_per_time_step_for_list:
            inc_out = increment_time(ind_rnap_list = [], ind_rnap_list_off = [],
                                     gene_map = gene_map, 
                                     par_list = par_list, 
                                     on_rate_per_time_step_rev = i, ## these values are iterated through 
                                     on_rate_per_time_step_for = j, ## these values are iterated through
                                     n_steps = n_steps, 
                                     bump_distance = bump_distance, 
                                     unidirectional = unidirectional)
            temp_df = get_sim_summary_df(inc_out)
            
            ## add simulation details to dataframe
            temp_df['for_on_rate'] = j
            temp_df['rev_on_rate'] = i
            temp_df['bump_distance'] = bump_distance
            temp_df['term_stay_prob'] = par_list['term_rev']['stay_pr']
            temp_df['term_for_prob'] = par_list['term_rev']['forward_pr']
            temp_df['term_rev_prob'] = par_list['term_rev']['release_pr']

            list_of_pd_df.append(temp_df)
            
    df_out = pd.concat(list_of_pd_df)
    
    return df_out
