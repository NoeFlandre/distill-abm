;;;;;; Agent Based Model of Food Choice Behaviour in the Context of British Milk Consumption ;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;; Agent's and global variables ;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


; attributes and model variables of each agent
turtles-own [
  ;mean values for the perceived characteristics, on a relative basis, of the milk choices
  incum-physical-mean
  incum-health-mean
  incum-env-mean
  alt-physical-mean
  alt-health-mean
  alt-env-mean

  ;milk choice characteristics perceived by agents, drawn from a normal distribution with mean values as above
  pphincum
  pinincum
  pexincum
  pphalt
  pinalt
  pexalt

  ;agent memory containing list of length determined by the memory-length parameter of the perceived charactersitics of each milk choice over time
  mem-incum-ph
  mem-alt-ph
  mem-incum-in
  mem-alt-in
  mem-incum-ex
  mem-alt-ex
  mem-ts

  ;values in the agent memory are averaged giving agent a single perception value for each milk choice and charactersitic
  mem-ph-incum-avg
  mem-ph-alt-avg
  mem-in-incum-avg
  mem-in-alt-avg
  mem-ex-incum-avg
  mem-ex-alt-avg

  ;the weighting applied to each of the three perception components - physcial charactersitics, health impact, environmental impact - in an agent's choice function
  ph-weight
  in-weight
  ex-weight
  ph-weight-raw
  in-weight-raw
  ex-weight-raw
  weights-raw

  ; choice functions for each milk option
  uf-incum
  uf-alt
  new-uf-incum
  new-uf-alt

  disposition-threshold             ;only for threshold-based model variant - float value between 0 and 1, above which an agent becomes disposed to consider its alternatives
  disposition                       ;float value between 0 and 1 of agent's disposition to consider its milk choice
  disposition-piqued?               ;binary TRUE or FALSE of whether agent disposition has been triggered or not
  f-red                             ;number of neighbours that consume mainly whole milk
  f-green                           ;number of neighbours that consume mainly skimmed/semi-skimmed milk
  f-all                             ;total number of neighbours
  h-entropy                         ;informational entropy of distribution of neighbours milk choice
  h-max                             ;maximum informational entropy of distribution of neighbours milk choice
  prob-disposition                  ;only or probability-based model variant - probability that an agent will become disposed to consider milk choices

  num-conseq-same-choice            ;number of consecutive same milk choices
  habit?                            ;variable to indicate whether habit function has been triggered
  habit-factor-incum                ;the factor applied to the whole milk choice function
  habit-factor-alt                  ;the factor applied to the skimmed and semi-skimmed milk choice function
  last-choice                       ;previous milk choice
  food-choice                       ;new milk choice
  habit-function                    ;choice function including the cognitve/perception score and habit factor

  conformity                        ;flat between 0 and 1 of degree to which agent conforms to the public concerns on health and environment
  incum-quantity                    ;amount in ml of weekly consumption of whole milk
  alt-quantity                      ;amount in ml of weekly consumption of skimmed and semi-skimmed milk
  prior-quantity-incum              ;amount in ml of weekly consumption of whole milk of previous time-step
  prior-quantity-alt                ;amount in ml of weekly consumption of skimmed and semi-skimmed milk of previous time-step
  choice-history                    ;agent record of previous choices
  incum-history                     ;agent record of previous whole milk choices
  alt-history                       ;agent record of previous skimmed/semi-skimmed choices

  sugar-imp                         ;weighted average amount of sugar from the combination of milk choice quantities
  satfat-imp                        ;weighted average amount of saturated fat from the combination of milk choice quantities
  protein-imp                       ;weighted average amount of protein from the combination of milk choice quantities
  co2-imp                           ;weighted average amount of CO2 from the combination of milk choice quantities
  land-imp                          ;weighted average amount of land requirement from the combination of milk choice quantities
  water-imp                         ;weighted average amount of water from the combination of milk choice quantities
  universalism                      ;integer between 0 and 6 for agent's score for basic human universalism value
  security                          ;integer between 0 and 6 for agent's score for basic human security value
  universalism-value                ;float between 0 and 1 for agent's score for basic human universalism value
  security-value                    ;float between 0 and 1 for agent's score for basic human security value
  openness                          ;integer between 0 and 6 for agent's score for basic human value associated with openness
  choice-value-health               ;float between 0 and 1 for the relative health impact of agent milk consumption choices
  choice-value-env                  ;float between 0 and 1 for the relative environmental impact of agent milk consumption choices
  choice-function-deviation         ;percentage difference between the maximum and minimum choice function scores
  value-health-deviation            ;the absolute difference between the held values of the agents, and the implied values by way of their choices
  value-env-deviation               ;the absolute difference between the held values of the agents, and the implied values by way of their choices
  cognitive-dissonance?             ;indicator for if agent is in a state of cognitive dissonance
  disposition-probability-gradient] ;only for probabilty-based disposition approach - the gradient, k, of the logisitic function governing the probability that an agent will become disposed to consider its milk consumption choices

;Global variables to run the model
globals [
  number-of-agents                  ;number of agents
  norm-data                         ;public concern data informed from Ipsos Mori and YouGov longitudinal survey data on concerns of the British public
  value-data                        ;data drawn from the UK results of three questions (assessing universalism, security, and openness) of the Human Values section of the European Social Survey 2018
  total-average-milk                ;
  min-unit                          ;the minimum allowable consumption of either milk choice, set at 1 pint (568ml)
  mean-incum                        ;main model output that measures the average whole milk consumption among agents
  mean-alt                          ;main model output that measures the average skimmed/semi-skimmed milk consumption among agents
  incum-total-try                   ;total instances of whole milk chosen by agents
  alt-total-try                     ;total instances of skimmed/semi-skimmed milk chosen by agents
  choice-total                      ;total choice made by agents

  ;variables governing the different health and environmental impacts associated with the milk choice
  sugar-list
  satfat-list
  protein-list
  co2-list
  land-list
  water-list
  sugar-realtive
  satfat-relative
  protein-relative
  co2-relative
  land-relative
  water-relative
  choice-health-sums
  choice-env-sums
  health-diff
  env-diff

  habit-on?                         ;habit function
  networks                          ;network function
  network-type                      ;type of network
  norms                             ;norms function
  counter]                          ;choice counter

; netlogo extension used in the model
extensions [
  csv ; reads csv files of data.
  nw  ; the network extension is used in this model.
  profiler]; assess model execution time

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;SETUP PROCEDURE;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to setup
  clear-all
  set number-of-agents 1000
  set mean-incum 2654.81             ;initialise average liquid wholemilk consumption per person per week (ml)
  set mean-alt 5.29                  ;initialise average skimmed (semi and full) consumption per person per week (ml)
  set habit-on? TRUE                 ;habit function
  set networks TRUE                  ;network function
  set network-type "watts-strogatz"  ;type of network
  set norms TRUE                     ;norm function
  ifelse networks = TRUE             ;creates a social network if networks is TRUE. Otherwise just creates a set of unconnected turtles.
    [create-network]
    [create-agents]
  setup-turtles
  setup-patches
  load-value-data                    ;loads 'Human value data' from UK response of ESS 2018
  assign-value-data                  ;agents are assigned values based on distribution from the ESS 2018 survey data
  file-close-all
  file-open "/Users/noeflandre/Documents/[03] School/École d'ingénieur/2A/Stage Miami University/Projet/Simulations/Comses/Historic British Milk Consumption ABM/data/netlogodata_downsampled.csv"             ;downsampled annual data on UK public concerns of economy, health, and environment - aggregated from Ipsos Mori Issues Index and YouGov.
  if justification < cognitive-dissonance-threshold
    [ifelse (justification + cognitive-dissonance-threshold) >= 1 [set justification 1] [set justification justification + cognitive-dissonance-threshold]]    ;makes sure that justification parameter is larger than cognitive-dissonance-threshold parameter during the optimisation exercise
  set network-parameter round network-parameter
  if remainder network-parameter 2 != 0 [set network-parameter network-parameter + 1] ;constrain agent neighbours to an even number
  impact-metrics
  reset-ticks
end

to setup-turtles
  foreach (list turtles) [[x] -> ask x [

  ;mean values as inputs to normal distribution to drawn values of milk charactersitics perceived by agents
  set incum-physical-mean 1
  set incum-health-mean 1
  set incum-env-mean 1
  set alt-physical-mean 1             ;for this part of the model, in comparing the development of skimmed versus whole milk, the physcial characterisitics of the alternative (skimmed/semi-skimmed) were fixed at 1 to explore the range of possible values taken by health and environmental perception.
  set alt-health-mean alt-health-mean-initial
  set alt-env-mean alt-env-mean-initial

  ;operating memory of agent's perception of milk choice characteristics
  set mem-incum-ph (list)
  set mem-alt-ph (list)
  set mem-incum-in (list)
  set mem-alt-in (list)
  set mem-incum-ex (list)
  set mem-alt-ex (list)
  set mem-ts (list)

  set last-choice color
  set food-choice color
  set disposition-piqued? FALSE
  set cognitive-dissonance? FALSE
  set habit? FALSE
  set habit-factor-incum 1
  set habit-factor-alt 1
  set num-conseq-same-choice incumbent-initial-habit    ;sets agent's consecutive prior choice equal to the incumbent-initial-habit parameter

  ;initital weighting of the three milk charactersitics perception components
  set ph-weight-raw random-float 1
  set in-weight-raw random-float 1
  set ex-weight-raw random-float 1

  ;initialise average liquid milk consumption per person per week (ml)
  set incum-quantity 2654.81
  set alt-quantity 5.29
  ]]

  ask turtles [set color red]
  agent-conformity                     ;runs the agent conformity function
end

to setup-patches
  ask patches [set pcolor blue]
  reset-ticks
end

to load-value-data
  ;this creates a distribution of values among the agents according to the ESS 2018 survey for the UK.
  ifelse (file-exists? "/Users/noeflandre/Documents/[03] School/École d'ingénieur/2A/Stage Miami University/Projet/Simulations/Comses/Historic British Milk Consumption ABM/data/netlogovaluedata.csv")
    [set value-data []
    set value-data (csv:from-file "/Users/noeflandre/Documents/[03] School/École d'ingénieur/2A/Stage Miami University/Projet/Simulations/Comses/Historic British Milk Consumption ABM/data/netlogovaluedata.csv")
    user-message "File loading complete!"
    file-close]
    [user-message "There is no netlogovaluedata.csv file in current directory!"]
end

to assign-value-data
  ;UK responses from three survey question from the European Social Survey (ESS) 2018 were selected that refer to one of three basic human values employed here: universalism, security, and openess.
  ;Universalism linked to pro-environmental and social attitudes and behaviours and used in the context of choice evaluation of environmental impacts
  ;Health falls within the Security basic human value. Used in context of choice evaluation of health impacts.
  ;Openness value used to set the disposition threshold of agents, above which they decide to consider alternatives.
  ifelse (is-list? value-data)
    [foreach value-data [four-tuple -> ask turtle item 1 four-tuple [set universalism item 2 four-tuple set security item 3 four-tuple set openness item 4 four-tuple]]]
    [user-message "You need to load in value data first!"]

  ;ESS value data given as 6-point Likert scale responses from "very much..." to "not at all...".
  ;These are operationalised for modelling purposes as a linear band with each point on the
  ;Likert-scale corresponding to 1/6 of the possible range of outputs from 0 to 1.
  ask turtles [
    if universalism = 0 [set universalism-value random-float 1]
    if universalism = 1 [set universalism-value ((5 / 6) + random-float (1 / 6))]
    if universalism = 2 [set universalism-value ((4 / 6) + random-float (1 / 6))]
    if universalism = 3 [set universalism-value ((3 / 6) + random-float (1 / 6))]
    if universalism = 4 [set universalism-value ((2 / 6) + random-float (1 / 6))]
    if universalism = 5 [set universalism-value ((1 / 6) + random-float (1 / 6))]
    if universalism = 6 [set universalism-value (0 + random-float (1 / 6))]

    if security = 0 [set security-value random-float 1]
    if security = 1 [set security-value ((5 / 6) + random-float (1 / 6))]
    if security = 2 [set security-value ((4 / 6) + random-float (1 / 6))]
    if security = 3 [set security-value ((3 / 6) + random-float (1 / 6))]
    if security = 4 [set security-value ((2 / 6) + random-float (1 / 6))]
    if security = 5 [set security-value ((1 / 6) + random-float (1 / 6))]
    if security = 6 [set security-value (0 + random-float (1 / 6))]

    if openness = 0 [set openness random-float 1]
    if openness = 6 [set disposition-threshold ((5 / 6) + random-float (1 / 6))]
    if openness = 5 [set disposition-threshold ((4 / 6) + random-float (1 / 6))]
    if openness = 4 [set disposition-threshold ((3 / 6) + random-float (1 / 6))]
    if openness = 3 [set disposition-threshold ((2 / 6) + random-float (1 / 6))]
    if openness = 2 [set disposition-threshold ((1 / 6) + random-float (1 / 6))]
    if openness = 1 [set disposition-threshold (0 + random-float (1 / 6))]]
end

to create-network
  if network-type = "watts-strogatz" [;ceates a Watts-Strogatz small-world network
    nw:generate-watts-strogatz turtles links number-of-agents network-parameter 0.1 [ fd 10 ]
    ask turtles [layout-spring turtles links 0.2 4.0 500]
  ]
end

to create-agents ;creates number-of-agents turtles if networks is FALSE (i.e., in case the model has no social networks).
  create-turtles number-of-agents
  [setxy random-xcor random-ycor]
end



to agent-conformity
  ; function to assign agent conformity, selected from a normal distribution with the mean equal to the social-conformity parameter
  if norms = TRUE [
    ask turtles [
      let mmin -1
      let mmax 1
      set conformity random-normal social-conformity 0.4
      if conformity < mmin or conformity > mmax
        [set conformity random-normal social-conformity 0.4]]
  ]
end


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;; GO PROCEDURE ;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


to go
  ;profiler:start
  if ticks >= 32 [stop]
  if file-at-end? [stop]
  set norm-data (csv:from-row file-read-line ",")
  set total-average-milk item 5 norm-data
  vary-info
  memory-fill
  memory-delete
  disposition-function
  cognitive-function
  social-influence
  habit-activation
  make-choice
  habit-formation
  prior-choice
  social-norms
  average-consumption
  ever-tried
  impact-tracker
  choice-evaluation
  prior-milk-amount
  ;if ticks >= 32 [profiler:stop]
  tick
end

to vary-info
  ;this function varies the information on milk characteristics perceived by agents
  ask turtles [
    if random-float 1 >= social-blindness and random-float 1 > 0.1 and ticks > 1 [
      if alt-quantity = 0 [
        ifelse (incum-health-mean / alt-health-mean) > 1000000
          [set alt-health-mean (alt-health-mean + (alt-health-mean * 0.05))] ;increases alt-health-mean by 5%
          [set alt-health-mean (alt-health-mean - (alt-health-mean * 0.05))] ;decreases alt-health-mean by 5%

        ifelse (incum-env-mean / alt-env-mean) > 1000000
          [set alt-env-mean (alt-env-mean + (alt-env-mean * 0.05))] ;increases alt-env-mean by 5%
          [set alt-env-mean (alt-env-mean - (alt-env-mean * 0.05))]] ;decreases alt-env-mean by 5%

      if incum-quantity = 0 [
        ifelse (incum-health-mean / alt-health-mean) > 0
          [set alt-health-mean (alt-health-mean + (alt-health-mean * 0.05))] ;increases alt-health-mean by 5%
          [set alt-health-mean (alt-health-mean - (alt-health-mean * 0.05))] ;decreases alt-health-mean by 5%

        ifelse (incum-env-mean / alt-env-mean) > 0
          [set alt-env-mean (alt-env-mean + (alt-env-mean * 0.05))] ;increases alt-env-mean by 5%
          [set alt-env-mean (alt-env-mean - (alt-env-mean * 0.05))] ;decreases alt-env-mean by 5%

      if incum-quantity != 0 and alt-quantity != 0 [
        ifelse (incum-health-mean / alt-health-mean) > ((incum-quantity * item 0 choice-health-sums) / (alt-quantity * item 1 choice-health-sums))
          [set alt-health-mean (alt-health-mean + (alt-health-mean * 0.05))] ;increases alt-health-mean by 5%
          [set alt-health-mean (alt-health-mean - (alt-health-mean * 0.05))] ;decreases alt-health-mean by 5%

        ifelse (incum-env-mean / alt-env-mean) > ((incum-quantity * item 0 choice-env-sums) / (alt-quantity * item 1 choice-env-sums))
          [set alt-env-mean (alt-env-mean + (alt-env-mean * 0.05))] ;increases alt-env-mean by 5%
          [set alt-env-mean (alt-env-mean - (alt-env-mean * 0.05))] ;decreases alt-env-mean by 5%
  ]]]

      let mmin 0.1

      set pphincum random-normal incum-physical-mean 0.1
      set pinincum random-normal incum-health-mean 0.1
      set pexincum random-normal incum-env-mean 0.1

      set pphalt random-normal alt-physical-mean 0.1
      set pinalt random-normal alt-health-mean 0.1
      set pexalt random-normal alt-env-mean 0.1

      while [pphincum <= mmin] [set pphincum random-normal incum-physical-mean 0.1]
      while [pinincum <= mmin] [set pinincum random-normal incum-health-mean 0.1]
      while [pexincum <= mmin] [set pexincum random-normal incum-env-mean 0.1]

      while [pphalt <= mmin] [set pphalt random-normal alt-physical-mean 0.1]
      while [pinalt <= mmin] [set pinalt random-normal alt-health-mean 0.1]
      while [pexalt <= mmin] [set pexalt random-normal alt-env-mean 0.1]
  ]
end

to memory-fill
  ;this function replicates agent memory, creating a list of information perceived by agents at each time step
  ask turtles [
    set mem-incum-ph lput pphincum mem-incum-ph
    set mem-alt-ph lput pphalt mem-alt-ph
    set mem-incum-in lput pinincum mem-incum-in
    set mem-alt-in lput pinalt mem-alt-in
    set mem-incum-ex lput pexincum mem-incum-ex
    set mem-alt-ex lput pexalt mem-alt-ex
    set mem-ts lput ticks mem-ts
  ]
end

to memory-delete
  ;this function replicates the finite nature of memory, capping the size of list containing perceived milk characteristics to equal the memory-lifetime parameter
  ask turtles [
    (foreach mem-incum-ph mem-alt-ph mem-incum-in mem-alt-in mem-incum-ex mem-alt-ex mem-ts
      [if (ticks - (first mem-ts)) >= memory-lifetime [
          let remove-position position first mem-ts mem-ts
          set mem-incum-ph remove-item remove-position mem-incum-ph
          set mem-alt-ph remove-item remove-position mem-alt-ph
          set mem-incum-in remove-item remove-position mem-incum-in
          set mem-alt-in remove-item remove-position mem-alt-in
          set mem-incum-ex remove-item remove-position mem-incum-ex
          set mem-alt-ex remove-item remove-position mem-alt-ex
          set mem-ts remove-item remove-position mem-ts]])
  ]
end

to disposition-function
  ;this function calculates and manages the process of agent disposition in the threshold-based model variant
  ask turtles [
    ifelse (count link-neighbors with [color != red]) = 0
      [set disposition 0]
      [ifelse (count link-neighbors with [color != green]) = 0
        [set disposition 0]
        [set disposition ((count link-neighbors with [color != red]) / (count link-neighbors with [color = red]))]]
    ifelse disposition >= disposition-threshold
      [set disposition-piqued? TRUE]
      [set disposition-piqued? FALSE]

    ;agents can also become disposed to consider alternatives given a state of cognitive dissonance
    if cognitive-dissonance? = TRUE [
    if choice-function-deviation = min(list choice-function-deviation value-health-deviation value-env-deviation)
      [set disposition-piqued? TRUE]]

    ;a small random of agents become spontaneously disposed
    if (disposition-piqued? = FALSE) and (random-float 1 >= .97) [set disposition-piqued? TRUE]
  ]
end

to disposition-function-probability
  ;this function calculates and manages the process of agent disposition in the probability-based model variant
  ask turtles [
    set f-red count link-neighbors with [color = red]
    set f-green count link-neighbors with [color = green]
    set f-all count link-neighbors
    set h-max (2 * (-(1 / 2) * log (1 / 2) 2))

    ifelse ((count link-neighbors with [color != red]) = 0) or ((count link-neighbors with [color != green]) = 0)
      [set h-entropy 0]
      [set h-entropy ((-(f-red / f-all) * log (f-red / f-all) 2) + (-(f-green / f-all) * log (f-green / f-all) 2))]

    set prob-disposition (1 / (1 + exp(- (disposition-probability-gradient) * ((h-entropy / h-max) - 0.5))))

    ifelse random-float 1 <= prob-disposition
      [set disposition-piqued? TRUE]
      [set disposition-piqued? FALSE]

    ;agents can also become disposed to consider alternatives given a state of cognitive dissonance
    if cognitive-dissonance? = TRUE [
    if choice-function-deviation = min(list choice-function-deviation value-health-deviation value-env-deviation)
      [set disposition-piqued? TRUE]]
  ]
end

to cognitive-function
  ;this function generates an agent's base score of the cognitive perception of the milk characterisitcs based on information
  ;it is exposed to, and the weights they ascribe to each of these characteristics
  ask turtles [
    set mem-ph-incum-avg mean mem-incum-ph
    set mem-ph-alt-avg mean mem-alt-ph
    set mem-in-incum-avg mean mem-incum-in
    set mem-in-alt-avg mean mem-alt-in
    set mem-ex-incum-avg mean mem-incum-ex
    set mem-ex-alt-avg mean mem-alt-ex
    set weights-raw (list ph-weight-raw in-weight-raw ex-weight-raw)
    set ph-weight ph-weight-raw / sum weights-raw
    set in-weight in-weight-raw / sum weights-raw
    set ex-weight ex-weight-raw / sum weights-raw
    set uf-incum (ph-weight * mem-ph-incum-avg + in-weight * mem-in-incum-avg + ex-weight * mem-ex-incum-avg)
    set uf-alt (ph-weight * mem-ph-alt-avg + in-weight * mem-in-alt-avg + ex-weight * mem-ex-alt-avg)
  ]
end

to social-influence
  ;this function represents peer influence, modelled by modifying an agent's cognitive milk choice function by the mean value among
  ;its neighbours with the effect size based on the social susceptibility parameter
  if networks = TRUE [
  ask turtles [
    if random-float 1 <= p-interact [
    ifelse count my-links >= 1
      [let ME self
      set new-uf-incum (([uf-incum] of ME) * (1 - social-susceptibility)) + ((sum [uf-incum] of link-neighbors) / (count my-links)) * (social-susceptibility)
      set new-uf-alt (([uf-alt] of ME) * (1 - social-susceptibility)) + ((sum [uf-alt] of link-neighbors) / (count my-links)) * (social-susceptibility)]
      [set new-uf-incum uf-incum
      set new-uf-alt uf-alt]
    set uf-incum new-uf-incum
    set uf-alt new-uf-alt]]
  ]
end

to social-norms
 ;this function represents the effect on how agents weight their cognitive perception by globally perceived public concerns on health and environemtnal issues.
 ;If norms are turned on then agents seek to conform (or not) to the prevailing social norms,
 ;modelled here as exogenous survey data (of issues/puvlic concerns index from Ipsos Mori and YouGov), operationlised as weightings between choice factors
 if norms = TRUE [
 ask turtles [
   let signs [-1 1]
   if ph-weight < item 1 norm-data [set ph-weight ph-weight + conformity / 100]
   if ph-weight > item 1 norm-data [set ph-weight ph-weight - conformity / 100]
   if conformity >= 0 and ph-weight = item 1 norm-data [set ph-weight ph-weight]
   if conformity < 0 and ph-weight = item 1 norm-data [set ph-weight (ph-weight + conformity / 100) * one-of signs]

   if in-weight < item 2 norm-data [set in-weight in-weight + conformity / 100]
   if in-weight > item 2 norm-data [set in-weight in-weight - conformity / 100]
   if conformity >= 0 and in-weight = item 2 norm-data [set in-weight in-weight]
   if conformity < 0 and in-weight = item 2 norm-data [set in-weight (in-weight + conformity / 100) * one-of signs]

   if ex-weight < item 3 norm-data [set ex-weight ex-weight + conformity / 100]
   if ex-weight > item 3 norm-data [set ex-weight ex-weight - conformity / 100]
   if conformity >= 0 and ex-weight = item 3 norm-data [set ex-weight ex-weight]
   if conformity < 0 and ex-weight = item 3 norm-data [set ex-weight (ex-weight + conformity / 100) * one-of signs]

   let weights (list ph-weight in-weight ex-weight)
   set ph-weight ph-weight / sum weights
   set in-weight in-weight / sum weights
   set ex-weight ex-weight / sum weights]
  ]
end

to habit-activation
  ;this function models whether an agent's choice is influenced by habit, and the size of this influence.
  ask turtles [
    let peak-habit 2
    if num-conseq-same-choice >= habit-threshold [set habit? TRUE]
    if habit-on? and habit? [set habit-function TRUE]
    if (habit-function = TRUE) and (food-choice = green)
      [set habit-factor-alt (peak-habit - exp (-0.042 * (num-conseq-same-choice - habit-threshold)))
      set habit-factor-incum 1]
    if (habit-function = TRUE) and (food-choice = red)
      [set habit-factor-incum (peak-habit - exp (-0.042 * (num-conseq-same-choice - habit-threshold)))
      set habit-factor-alt 1]
    if habit-function = FALSE [set habit-factor-incum 1 set habit-factor-alt 1]
  ]
end

to make-choice
  ;this function represents the main decision making function where the cognitive functions of milk choices, modifed by social effects and habit,
  ;are compared and milk consumption is assigned proportionately to the respective size of these scored functions.
  ask turtles [
    ifelse (disposition-piqued? = TRUE) [
      if ((uf-incum * habit-factor-incum) > (uf-alt * habit-factor-alt)) [set color red set food-choice red]
      if ((uf-alt * habit-factor-alt) > (uf-incum * habit-factor-incum)) [set color green set food-choice green]
      set choice-history choice-history + 1
      if food-choice = red [set incum-history incum-history + 1]
      if food-choice = green [set alt-history alt-history + 1]]

      [set color color set food-choice last-choice
      set choice-history choice-history + 1
      if food-choice = red [set incum-history incum-history + 1]
      if food-choice = green [set alt-history alt-history + 1]]

    set min-unit 568 ;ml of 1 British pint

    if (disposition-piqued? = TRUE) and ((uf-incum * habit-factor-incum + uf-alt * habit-factor-alt) != 0) [
      set incum-quantity ((uf-incum * habit-factor-incum) / (uf-incum * habit-factor-incum + uf-alt * habit-factor-alt)) * total-average-milk
      set alt-quantity ((uf-alt * habit-factor-alt) / (uf-incum * habit-factor-incum + uf-alt * habit-factor-alt)) * total-average-milk
    if incum-quantity < min-unit [
      set alt-quantity alt-quantity + incum-quantity
      set incum-quantity 0]
    if alt-quantity < min-unit [
      set incum-quantity incum-quantity + alt-quantity
      set alt-quantity 0]]

    ifelse (disposition-piqued? = FALSE) and ticks > 1[
      set incum-quantity prior-quantity-incum set alt-quantity prior-quantity-alt]
      [set incum-quantity incum-quantity set alt-quantity alt-quantity]
 ]
end

to ever-tried
  ;this function tracks the total number of choices for each milk type
  set incum-total-try sum [incum-history] of turtles
  set alt-total-try sum [alt-history] of turtles
  set choice-total sum [choice-history] of turtles
end

to average-consumption
  ;this function tracks the mean consumption (ml/week) of each milk type across agents
  if ticks < 1 [set mean-incum 2654.81 set mean-alt 5.29]
  if ticks >= 1 [set mean-incum mean [incum-quantity] of turtles set mean-alt mean [alt-quantity] of turtles]
end

to habit-formation
  ;this function tracks the number of consecutive same choices by agents, and informs the habit function
  ask turtles [
  ifelse food-choice = last-choice
    [set num-conseq-same-choice num-conseq-same-choice + 1]
    [set num-conseq-same-choice 0]
  ]
end

to prior-choice
  ;this function manages agent's prior and current choices
  ask turtles [
    if last-choice != food-choice [set counter counter + 1]
    set last-choice food-choice
  ]
end

to prior-milk-amount
  ;this function manages agent's prior and current milk choice quantities
  ask turtles [
    set prior-quantity-incum incum-quantity set prior-quantity-alt alt-quantity
  ]
end

to impact-metrics
  ;this function contains data on health and environmental impact metrics. The first values in each list refer to whole milk, the latter refer to skimmed/semi-skimmed.
  ;Values are per litre. CO2 impact is British Isles (BI) specific and differentiated by whole or semi/skimmed. Land and water are not BI specific or differentiated.
  set sugar-list [49.39 50.62] ;grams
  set satfat-list [19.76 6.61] ;grams
  set protein-list [36.12 37.03] ;grams
  set co2-list [1.30 1.07] ;kgCO2e
  set land-list [9 9] ;m2
  set water-list [628 628] ;litres
  set sugar-realtive [0.98 1.00] ;
  set satfat-relative [1.00 0.33] ;
  set protein-relative [0.98 1.00] ;
  set co2-relative [1.00 0.82] ;
  set land-relative [1.00 1.00] ;
  set water-relative [1.00 1.00] ;
  set choice-health-sums [1.00 0.33] ;note the protein score is subtracted from the health sum as more protein per serving is deemed beneficial
  set choice-env-sums [3.00 2.82] ;
  set health-diff (max(choice-health-sums) - min(choice-health-sums))
  set env-diff (max(choice-env-sums) - min(choice-env-sums))
end

to impact-tracker
  ;this function tracks the overall size of the health and environmental impact based on the quantities of each type of milk consumed by agents.
  ask turtles [
    set sugar-imp ((incum-quantity * item 0 sugar-list) + (alt-quantity * item 1 sugar-list)) / 1000
    set satfat-imp ((incum-quantity * item 0 satfat-list) + (alt-quantity * item 1 satfat-list)) / 1000
    set protein-imp ((incum-quantity * item 0 protein-list) + (alt-quantity * item 1 protein-list)) / 1000
    set co2-imp ((incum-quantity * item 0 co2-list) + (alt-quantity * item 1 co2-list)) / 1000
    set land-imp ((incum-quantity * item 0 land-list) + (alt-quantity * item 1 land-list)) / 1000
    set water-imp ((incum-quantity * item 0 water-list) + (alt-quantity * item 1 water-list)) / 1000
  ]
end

to choice-evaluation
  ;this function models the evaluation of an agent's choice against its human values, and determines if an agent will enter a state of cognitive dissonace.
  ask turtles [
    if random-float 1 >= social-blindness [
      let incumbent-choice-function (uf-incum * habit-factor-incum)
      let alterative-choice-function (uf-alt * habit-factor-alt)
      let weighted-average-health-impact (incum-quantity * item 0 choice-health-sums + alt-quantity * item 1 choice-health-sums) / total-average-milk
      let weighted-average-env-impact (incum-quantity * item 0 choice-env-sums + alt-quantity * item 1 choice-env-sums) / total-average-milk

      set choice-value-health weighted-average-health-impact - min(choice-health-sums) * (1 / (health-diff))
      set choice-value-env weighted-average-env-impact - min(choice-env-sums) * (1 / (env-diff))

      set value-health-deviation abs (choice-value-health - security-value)
      set value-env-deviation abs (choice-value-env - universalism-value)

      ; choice-function-deviation calculates the size of the difference, in percentage terms, between the highest scored choice and he mean of the other scores.
      set choice-function-deviation (max( list incumbent-choice-function alterative-choice-function) - (sum (list incumbent-choice-function alterative-choice-function) - (max( list incumbent-choice-function alterative-choice-function))) / 2) / (max( list incumbent-choice-function alterative-choice-function))
      ifelse ((value-health-deviation >= cognitive-dissonance-threshold) and (value-health-deviation <= justification)) or ((value-env-deviation >= cognitive-dissonance-threshold) and (value-env-deviation <= justification))
        [set cognitive-dissonance? TRUE]
        [set cognitive-dissonance? FALSE]

      if cognitive-dissonance? = TRUE [
        if (value-health-deviation = min(list choice-function-deviation value-health-deviation value-env-deviation)) and (count my-links >= 1)
          [let ME self
          set security-value (([security-value] of ME) * ( 1 - social-susceptibility )) + ((sum [security-value] of link-neighbors) / ( count my-links )) * (social-susceptibility)]
        if (value-env-deviation = min(list choice-function-deviation value-health-deviation value-env-deviation)) and (count my-links >= 1)
          [let ME self
          set universalism-value (([universalism-value] of ME) * ( 1 - social-susceptibility )) + ((sum [universalism-value] of link-neighbors) / ( count my-links )) * (social-susceptibility)]]]
  ]
end
@#$#@#$#@
GRAPHICS-WINDOW
424
21
941
539
-1
-1
9.9804
1
10
1
1
1
0
1
1
1
-25
25
-25
25
1
1
1
ticks
30.0

BUTTON
28
36
94
69
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
28
81
91
114
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

MONITOR
30
141
118
186
Red turtle
count turtles with [color = red]
17
1
11

MONITOR
30
202
117
247
Green turtle
count turtles with [color = green]
17
1
11

PLOT
27
258
411
420
Milk choice
Time
Total
0.0
100.0
0.0
100.0
true
true
"" ""
PENS
"Incumbent" 1.0 0 -2674135 true "" "plot (count turtles with [color = red]) / (count turtles) * 100"
"Alternative" 1.0 0 -13840069 true "" "plot (count turtles with [color = green]) / (count turtles) * 100"

SLIDER
963
24
1145
57
memory-lifetime
memory-lifetime
1
10
5.0
1
1
NIL
HORIZONTAL

SLIDER
964
67
1161
100
alt-health-mean-initial
alt-health-mean-initial
1
3
1.69
0.01
1
NIL
HORIZONTAL

SLIDER
970
215
1142
248
habit-threshold
habit-threshold
1
10
2.0
1
1
NIL
HORIZONTAL

SLIDER
970
359
1142
392
p-interact
p-interact
0
1
0.3
0.1
1
NIL
HORIZONTAL

SLIDER
970
401
1143
434
social-susceptibility
social-susceptibility
0
1
0.2
0.1
1
NIL
HORIZONTAL

SLIDER
970
257
1143
290
incumbent-initial-habit
incumbent-initial-habit
0
10
10.0
1
1
NIL
HORIZONTAL

SLIDER
970
532
1144
565
social-conformity
social-conformity
-1
1
0.28
0.01
1
NIL
HORIZONTAL

SLIDER
964
111
1146
144
alt-env-mean-initial
alt-env-mean-initial
0
3
1.12
0.01
1
NIL
HORIZONTAL

PLOT
19
438
417
633
Average consumption (ml per week)
Time (years)
Consumption per person/HH (ml/week)
0.0
32.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -2674135 true "" "plot mean-incum"
"pen-1" 1.0 0 -13840069 true "" "plot mean-alt"

SLIDER
969
490
1142
523
social-blindness
social-blindness
0
1
0.51
0.01
1
NIL
HORIZONTAL

SLIDER
1235
529
1463
562
justification
justification
0
1
0.61
0.01
1
NIL
HORIZONTAL

SLIDER
1234
486
1463
519
cognitive-dissonance-threshold
cognitive-dissonance-threshold
0
1
0.15
0.01
1
NIL
HORIZONTAL

SLIDER
969
446
1143
479
network-parameter
network-parameter
2
10
8.0
0.01
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

The overall objective of the agent-based model is to reproduce adoption behaviours of milk consumption by the British public (1974-2005), by replicating individual preferences and decision factors. Specifically, the goal of the model is to explore the influence of perception, habits and social influence in an individual’s decision-making process of milk choice. The model looks to replicate the substitution of whole milk for skimmed and semi-skimmed milk from the 1970s onwards. The main outcome reported here is the average weekly consumption of each milk type per person. The simulation uses both a theoretical grounding and empirical data to inform the ABM, with calibration performed against observed macro level data.

In particular, the study conducts an experiment to compare the performance of two model variants in reproducing overserved milk consumption trends. These variants present different mechanisms for how agents become disposed to consider their choices, representing a threshold based, and a probability-based approach.


## HOW IT WORKS

Decision-making follows a basic structure of: agent perception of choice characteristics, the triggering, or not, of disposition to consider alternatives, a set of scored choice functions made up of the perceived characteristics and modulated by habit and social influence, and finally, choice evaluation where agents consider the impact of their choices and may adjust their future decisions.

The agents in the model represent adult consumers who occupy a random position in an information environment. Each agent has a disposition to consider alternative milk choices. Two disposition mechanisms are tested in the model, a threshold-based approach, and a probability-based approach. Each agent makes a choice of milk selection based on a function for each alternative, made up of the perceived health and environmental characterises of each choice. These are computed at the initialisation of the simulation and then calculated at each time step. An agent’s milk choice function is modified by habit, social influence, and evaluation of previous choices. Agents ascribe different relative importance to each constituent part of the choice function (health factors and environmental factors). If the disposition requirement has been met, consumption of each milk type is split proportionately by the size of each choice function, modulated by the other influences. If not, agents keep their existing choice.

Agents (n=1,000) start with an existing choice based on the dominant position of whole milk versus skimmed varieties in 1974 (start year of the data). All agents are part of a social network. Each agent in the network can sense and be influenced by the choice function of each milk alternative for other agents in their network. Links between agents are unidirectional, and influence occurs as a function of interaction probability, with the degree of influence characterised by agent susceptibility. Social norms are globally perceived by agents and impact the weightings of the choice function


## HOW TO USE IT

Setup and Go.

Monitors:

Two monitors track the number of turtles that have consume mostly whole-milk (incumbent initial choice) or skimmed/semi-skimmed milk (alternative initial choice).

Plots:

One plot shows the percentage of majority whole milk or skimmed/semi-skimmed milk consumers.

The second plots the average consumption of each milk type over time. This is the data that the calibration exercise uses to try and replicate observed consmption data.

Parameters:

12 paraemters that governs whether an agent will consider changing its milk type, and how much of each milk type an agent will consume. We refer viewers to the manuscript based on this model for further details.

## THINGS TO NOTICE

The appearance of a crossover in the types of milk consumed and the shape of the curves over time.

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="experiment" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>count turtles</metric>
    <enumeratedValueSet variable="alt-env-mean-initial">
      <value value="1.12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="network-parameter">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-lifetime">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="habit-threshold">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cognitive-dissonance-threshold">
      <value value="0.15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="p-interact">
      <value value="0.3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-blindness">
      <value value="0.51"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-conformity">
      <value value="0.28"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-susceptibility">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="alt-health-mean-initial">
      <value value="1.69"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="incumbent-initial-habit">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="justification">
      <value value="0.61"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Milk Consumption Trends" repetitions="30" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <exitCondition>ticks = 100</exitCondition>
    <metric>mean-incum</metric>
    <metric>mean-alt</metric>
    <metric>incum-total-try</metric>
    <metric>alt-total-try</metric>
    <metric>choice-total</metric>
    <metric>mean [incum-quantity] of turtles</metric>
    <metric>mean [alt-quantity] of turtles</metric>
    <metric>mean [habit-factor-incum] of turtles</metric>
    <metric>mean [habit-factor-alt] of turtles</metric>
    <metric>mean [choice-function-deviation] of turtles</metric>
    <metric>mean [value-health-deviation] of turtles</metric>
    <metric>mean [value-env-deviation] of turtles</metric>
    <enumeratedValueSet variable="number-of-agents">
      <value value="1000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="habit-on?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="network-type">
      <value value="&quot;watts-strogatz&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="norms">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="network-parameter">
      <value value="8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="memory-lifetime">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="habit-threshold">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="cognitive-dissonance-threshold">
      <value value="0.15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="p-interact">
      <value value="0.3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-blindness">
      <value value="0.51"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-conformity">
      <value value="0.28"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-susceptibility">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="alt-health-mean-initial">
      <value value="1.69"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="alt-env-mean-initial">
      <value value="1.12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="incumbent-initial-habit">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="justification">
      <value value="0.61"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
