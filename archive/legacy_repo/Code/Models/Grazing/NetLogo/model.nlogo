;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; RAGE RAngeland Grazing ModEl ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; comments with #EXT tag relate to the model extension with learning processes/agents
;; #EXT include the array extension
extensions [array]

;; part of the base model (vegetation and livestock growth) and the plotting and output functions are moved to external source files
;; you can have a look at them if you're curious but you probably won't need to modify them
__includes [ "RAGE_VegetationLivestockModel.nls" "RAGE_PlottingOutput.nls" ]



globals
[
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; model parameters
  ;; variable names that are commented are defined via the interface, only noted here for completeness
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;gr1                                ; grazing pressure on green biomass
  ;gr2                                ; grazing pressure on reserve biomass
  ;w                                  ; biomass growth rate
  rue                                 ; rain use efficiency
  mg                                  ; mortality of green biomass
  mr                                  ; mortality of reserve biomass
  lambda                              ; growth limit of green biomass
  Rmax                                ; maximum reserve biomass
  d                                   ; density dependence of reserve biomass, d = 1/Rmax
  R0part                              ; initial reserve biomass as fraction of Rmax
  green-biomass-carry-over?           ; should green biomass over be carried over to the next year?
  ;rain-mean                          ; average rainfall in [mm]
  ;rain-std                           ; standard deviation of rainfall
  ;b                                  ; sheep birth rate
  intake                              ; sheep fodder intake
  global-rain                         ; rainfall value for all pastures when global rainfall is used
  ;number-households                  ; number of households in the system at the beginning
  number-pastures                     ; number of pastures in the system
  ;timesteps                          ; number of ticks that the simulation will be run
  ;start-on-homepatch?                ; should the households start each tick on their initial (home) patch, or their current one?
  ;resting?                           ; should resting be used?
  ;knowledge-radius                   ; knowledge radius of the households
  ;minimum-viable-herd-size           ; minimum viable herds size that households need to fulfil their livelihood
  ;behavioral-type                    ; decision model of the households
  ;satisficing-threshold              ; behavioral parameter
  ;satisficing-trials                 ; behavioral parameter
  local-first?                        ; behavioral parameter
  descriptive-norm
  use-rain-from-file?
  global-rainfall-list
  ;extension-model?                   ; #EXT is the extension model running or the original one?
  ;SL2-strategy-switch?               ; #EXT turns global strategy switching on/off?
  ;risk-mode?                         ; #EXT enables agents' heterogeneous behaviour based on their r-parameter (ONLY implemented for E-LBD type for experimental purposes; could be extended to E-RO-SL1; E-RO with this parameter OFF would be equivalent to MAX without movement)
  ;livestock-init                     ; #EXT set this in the interface for testing purposes while developing extension-model, it was a fixed number in the original model
  extension-model-behaviors-list      ; #EXT list of behaviors added in the extension model

  ;; other variables
  sum-livestock-total
  model-seed
  household-strategy-counts           ; list of mixed strategy counts

  vid-step
]

;; household turtles
breed [ households household ]

patches-own
[
  green-biomass-init ; green biomass at the beginning of the tick, before grazing
  green-biomass ; actual available green biomass, i.e. possibly diminshed due to grazing
  reserve-biomass
  reserve-biomass-edible ; part of reserve biomass that can be consumed in the current tick
  rain
  is-being-grazed?
  patch-available-for-grazing? ; NH patch-available-for-grazing? = TRUE IFF is-rested? = TRUE; Todo: delete variable

  ;; variables for policy: RESTING
  is-rested? ;NH todo delete, use "consecutive-years-rested" instead
  consecutive-years-grazed
  consecutive-years-rested
]

households-own
[

  livestock

  destock
  homepatch

  ;; households strategy parameters
  household-behavioral-type                   ; which behavioral type does the household follow (e.g. one of TRAD, MAX, SAT #EXT added E-LBD type)
  household-knowledge-radius                  ; radius of perception of pastures around own location
  household-local-neighborhood

  household-intrinsic-preference              ; intrinsic preference for pasture resting
  household-social-susceptibility             ; susceptibility to behavior of other households
  household-satisficing-threshold             ; satisficing threshold for herd size
  current-household-satisficing-threshold     ; gets updated if current herd size is lower than the actual satisficing threshold (e.g. due to destocking)
  household-effective-propensity              ; effective propensity of the household for resting / not resting, given its preferences and the descriptive norm
  household-resting-behavior                  ; 1 - household abides to resting rule, 0 - household does not abide to resting rule

  household-satisficing-trials                ; for SAT: number of pastures that the household evaluates when following type SAT (i.e. defines its cognitive limitation)
  household-local-first                       ; for SAT: should the household search in it's local neighborhood first or is distance irrelevant when evaluating patches?


  ;; #EXT variables for learning extension
  household-reserve-biomass-memory             ; used for E-LBD: reserve biomass seen in previous ticks; this is defined as an array type #EXT
  household-calc-diff                          ; used for E-LBD: calculate difference in observed reserve biomass and store here #EXT
  household-livestock-placed-memory            ; for extension agents: livestock placed on the pasture each round; this is defined as an array #EXT
  household-risk-att                           ; for extension agents: degree of deviation from no. of livestock recommended by policy (calc-diff), this is called "r-parameter" in documentation #EXT
  household-risk-att-init                      ; for extension agents E-SL: used to store and report (in Behavior Space) initial risk attitude of social learning agents
  household-behavioral-type-init               ; for extension agents with SL2 switching strategy on: used to store and report (in Behavior Space) initial behavioral type, before switching

  household-SL?                                ; used for E-SL: for social learning agents, this becomes true once they learned something (e.g. risk profile or switch of strategy); used for analysis
  household-SL2?                               ; used for overall SL2: when strategy switching global behaviour is on, variable becomes true once agent has switched strategy at least once; used for analysis
  new-household-behavioral-type                ; used for updating behav type at the beginning of following round
  new-livestock                                ; used for updating livestock no in SL1 procedure, after all neighbours have made their evaluations of successful neighbours
  new-household-risk-att                       ; used for updating risk att in SL1 procedure, after all neighbours have made their evaluations of what risk attitude they are learning from neighbours
  new-color                                    ; used for changing color according to new behavioral type when SL2 is on

  ;; variables to keep track of livestock / destock over time ;NH todo delete (this was original model comment)
  livestock-healthy-total                     ; used in #EXT model to count total of livestock-healthy over time
  destock-total                               ; used in #EXT model to count total livestock that went hungry over time
]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; MODEL SETUP ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to setup
  clear-ticks
  clear-turtles
  clear-patches
  clear-drawing
  clear-all-plots
  clear-output

  ;; set the model random seed
  set model-seed new-seed
  ;set model-seed 13911 ;#EXT fixed for testing
  random-seed model-seed

  ;; set model parameter values
  set rue 0.002
  set mg 0.1
  set mr 0.05
  set Rmax 150000
  set d 1 / Rmax
  set R0part 0.6
  set lambda 0.5
  set intake 640
  set number-pastures 100
  set local-first? true
  set green-biomass-carry-over? true

  set use-rain-from-file? false

  ;;#EXT define list of behaviors added in the extension model
  set extension-model-behaviors-list (list "E-RO" "E-LBD" "E-RO-SL1")


  if ( use-rain-from-file? )
  [
    ifelse ( file-exists? "rainfall_list.txt" )
    [
      file-open "rainfall_list.txt"
      set global-rainfall-list file-read
      file-close
    ]
    [
      user-message ( "No rainfall file found, create a rainfall file with generate-rain-list first!" )
    ]
  ]



  ;; #EXT in extension model, timestep should not exceed 500 because of limitations of agent memory;

  if (extension-model?)
  [
    if timesteps > 500
          [user-message ("With current settings, the model can only run up to 500 timesteps. Please first edit the code to increase agent memory length.")]
  ]


  ;; now create a number of pastures and initialize them
  ask patches
  [

  ;; #EXT for extension model set fixed value for rain
    ifelse ( extension-model? ) ;#EXT
       [set rain 282.71]        ;#EXT
       [set rain rainfall]

    set reserve-biomass R0part * Rmax
    set reserve-biomass-edible reserve-biomass * gr2
    set green-biomass-init lambda * reserve-biomass
    set green-biomass  lambda * reserve-biomass


    ;; per default, all patches are available for grazing
    set patch-available-for-grazing? true
    set is-being-grazed? false

    ;; variables for resting policy
    set is-rested? false
    set consecutive-years-grazed 0
    ;NH set consecutive-years-rested 0

    ;; layout and colors
    set pcolor scale-color green (reserve-biomass-edible + green-biomass) ( 73000 + gr2 * Rmax ) 0
    set plabel-color white
    set plabel ""
  ]

  ;; create a number of households and initialize them; #EXT initialize new agent properties in extension-model
  ;; households are initally placed on a random pasture where no other households are present

  if (number-households > 100)
  [
    user-message "Not more than 100 households are allowed - households are automatically set to 100"
    set number-households 100
  ]

  ask n-of number-households patches
  [
    sprout-households 1
    [
      set homepatch patch-here
      set shape "house4"
      set livestock livestock-init ;#EXT original was 10, changed to a param in interface.
      set destock 0
      set pen-size 1
      set color red
      set label-color white
      ; behavioral types will be set up in separate procedure
      set household-behavioral-type "none"
      ; adjust the size of the household shape according to their current number of livestock
      set size max list 0.25 (livestock / 120)
      ; #EXT initialize variables for global behaviour strategy switching (SL2)
      set new-household-behavioral-type "none"
      set new-household-risk-att "none"
      set new-color "none"
      ; #EXT initialize variables to track economic performance
      set livestock-healthy-total 0
      set destock-total 0
    ]
  ]

  ; for behavior space
  if homog-behav-types?
  [
    ask households
    [
      set household-behavioral-type behavioral-type
    ]
  ]


  ;; setup household relocation strategies
  setup-household-behavioral-types

  ; #EXT for all extension model agents, initialize memory vectors - reserve biomass, livestock placed; also initialize variable for storing calculation
  if extension-model?
  [
    ask households
    [ if (member? household-behavioral-type extension-model-behaviors-list)

      [
        ;#EXT create empty vector
        set household-reserve-biomass-memory array:from-list n-values 501 [0]
        ;# memory of reserve biomass is initialized with initial amount of reserve-biomass (see variable reserve-biomass = R0part * Rmax set above); all patches have the same initial reserve-biomass
        array:set household-reserve-biomass-memory 0 [reserve-biomass] of patch-here
        set household-calc-diff 0
        set household-livestock-placed-memory array:from-list n-values 501 [0]
        ;#EXT memory initialized with livestock variable - at this time this holds the init-livestock value (from setup)
        array:set household-livestock-placed-memory 0 livestock


        ;for BehaviorSpace exp New-one-agent-sensitivity -> vary household-risk-att as an input param from interface (risk-att-init) with values between -0.95 and 0.95
        ;set household-risk-att risk-att-init

        ; changed this to interval -0.95 and 0.95
        let i random 39 / 20
        set i (i - 0.95)
        ; print (word "this house has risk: " i)
        set i (precision i 2)
       ; print (word "rounded risk is: " i)
        set household-risk-att i
        set household-risk-att-init household-risk-att


        ifelse household-risk-att >= 0
        [ set shape "triangle"
        ]
        [set shape "triangle-down"
        ]
      ]

    ]
  ]



  set descriptive-norm ( count households with [ patch-available-for-grazing? = true ] ) / ( count households )

  reset-ticks
end



to setup-household-behavioral-types
  ;; function to setup individual strategies of households

  ;; determine if we're running a simulation with homogeneous or mixed behavioral types
  if not homog-behav-types?
  [
    ; heterogeneous types: numbers set via sliders in interface

;; #EXT for behavior space, calculate other strategies automatically based on sole input number-E-LBD; this constraint doesn't allow mixing of behaviour types from the original model. Variables below are input in Beh Sp param
  ;#EXT for running New-multiple-agents-SL-switch-ELBD - uncomment two lines below
     ;set number-E-RO (number-households - number-E-LBD) / 2
     ;set number-E-RO-SL1 number-E-RO
  ;#EXT for running New-multiple-agents-SL-switch-E-RO-SL1 - uncomment two lines below
     ;set number-E-RO (number-households - number-E-RO-SL1) / 2
     ;set number-E-LBD number-E-RO
  ;#EXT for running New-multiple-agents-SL-switch-eq (equal no of agents) - uncomment three lines below
     ;set number-E-RO (number-households / 3)
     ;set number-E-RO-SL1 number-E-RO
     ;set number-E-LBD number-E-RO

    set household-strategy-counts ( list number-random number-MAX number-SAT number-TRAD number-E-RO number-E-LBD number-E-RO-SL1) ;#EXT added new types
    let strategy-list ( list "Random" "MAX" "SAT" "TRAD" "E-RO" "E-LBD" "E-RO-SL1") ;#EXT added new types
    let curr-strategy 0

    foreach household-strategy-counts
    [ ?1 ->
      ask n-of ?1 households with [ household-behavioral-type = "none" ]
      [
        set household-behavioral-type item curr-strategy strategy-list
      ]
      set curr-strategy curr-strategy + 1
    ]
  ]

  ask households
  [
    ;NH all households have the same knowledge-radius
    set household-knowledge-radius knowledge-radius ;NH
    set household-local-neighborhood moore-neighborhood household-knowledge-radius ;NH

    ;#EXT all households start with household-SL? and -SL2? false
    set household-SL? FALSE
    set household-SL2? FALSE

    set household-behavioral-type-init household-behavioral-type

    if ( household-behavioral-type = "Random" )
    [
      set color yellow
      ; no parameters to set for this strategies
    ]

    if ( household-behavioral-type = "MAX" )
    [
      set color red
      ; MAX type - no preference for resting, no social susceptibility
      ;            satisficing threshold quasi infinity
      set household-intrinsic-preference 0.0
      set household-social-susceptibility 0.0
      set household-satisficing-threshold 9999
      set household-satisficing-trials 9999
      set household-local-first local-first?
      set household-resting-behavior 0
      ; no parameters to set for these strategies
    ]

    if ( household-behavioral-type = "SAT" )
    [
      set color blue
      ; SAT type - no preference for resting, no social susceptibility
      ;            set satisficing threshold accoring to sliders
      set household-intrinsic-preference 0.0
      set household-social-susceptibility 0.0
      set household-satisficing-threshold satisficing-threshold
      set household-satisficing-trials satisficing-trials
      set household-local-first local-first?
      set household-resting-behavior 0
    ]

    if ( household-behavioral-type = "TRAD" )
    [
      ; TRAD type - intrinsic preference and social susceptibility according to sliders
      ;             satisficing threshold quasi infinity
      set household-intrinsic-preference intrinsic-preference
      set household-social-susceptibility social-susceptibility
      set household-satisficing-threshold 9999
      set household-satisficing-trials 9999
      set household-local-first local-first?
      set household-resting-behavior 0
      set color scale-color violet household-intrinsic-preference 0 1
    ]
   ; #EXT new type for extension model w learning
   ;          all other parameters are the same as for MAX
    if (household-behavioral-type = "E-LBD")
    [ set color cyan
      set household-intrinsic-preference 0.0
      set household-social-susceptibility 0.0
      set household-satisficing-threshold 9999
      set household-satisficing-trials 9999
      set household-local-first local-first?
      set household-resting-behavior 0
    ]

    ; #EXT new type for extension model, risk only agents, they function as MAX agent and destock as normal
    ;         all other parameters are the same as for MAX
    if (household-behavioral-type = "E-RO")
    [ set color white
      set household-intrinsic-preference 0.0
      set household-social-susceptibility 0.0
      set household-satisficing-threshold 9999
      set household-satisficing-trials 9999
      set household-local-first local-first?
      set household-resting-behavior 0
    ]

    ; #EXT new type for extension model,
    ;         all other parameters are the same as for MAX
    if (household-behavioral-type = "E-RO-SL1")
    [ set color lime
      set household-intrinsic-preference 0.0
      set household-social-susceptibility 0.0
      set household-satisficing-threshold 9999
      set household-satisficing-trials 9999
      set household-local-first local-first?
      set household-resting-behavior 0
    ]
  ]
end


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; GO FUNCTION ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to go
  ;; check first if we have reached the end of the simulation
  if ( ticks > timesteps - 1)  [ stop ]

  ;; I'm using tick-advance instead of tick to avoid the automatic updating of the plots
  tick-advance 1

  ;print (" ")
  ;print (word "tick: " ticks)
  ;print (word "NEW ROUND STARTED")


  ;; #EXT for extension-model, original checks for rain and resting are no longer needed
  if not extension-model?
  [rain-and-resting]


;; patches ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ask patches
  [update-green-biomass-new-tick]


;; households ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

ask households [
  ifelse (not member? household-behavioral-type extension-model-behaviors-list)

; ORIGINAL AGENT TYPES SEQUENCE
    [
    ;; in original model livestock growth happens first;
      set livestock livestock-growth livestock
     ; print (word "who is: " who " livestock now is: " livestock " behavioural type is: " household-behavioral-type)  ;;; ### TEST this
      set destock 0

    ;; special behaviour for SAT type, from original model
      if household-behavioral-type = "SAT"
      [
          ; if household is of type SAT, he will destock to his satisficing threshold first before relocation
          destock-SAT
      ]
    ;; move back to homepatch, if start-on-homepatch is set to true
      if ( start-on-homepatch? ) [ move-to homepatch ]

    ;; relocate the households according to the choosen relocation strategy
      relocate-household

      livestock-feed

    ;; households with no livestock die in the original model
      if (livestock <= 0)
      [
         ;print ( word who ": died, probably because of hunger > reenters with 1 sheep" )
         die
      ]
     ] ;end of sequence for original agent types

; EXTENSION MODEL AGENTS TO GO
     [ ;; order of procedures for EXTENSION agents model
    ;print ("")
    ;print (word "HOUSEHOLD " who)

    ;; STEP 1A;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

     ; households with SL1 behaviour should also update their r param value with the one retained last round from successful neighbor (in the observe neighbours procedure)
      if (household-behavioral-type = "E-RO-SL1")
         [update-r-param]

     ; if strategy switch is on, update behavioral type with the one retained last round from successful neighbor; an update can only happen, thus, from tick 2 onward
      if (SL2-strategy-switch?)
         [update-household-strategy]





       ;print (word who " livestock-healthy: " livestock " destock: " destock)


     ;; STEP 1B;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
       ;; #EXT original model agents relocate their household on empty patches
       ;; #EXT for extension version agents, a different procedure is run: at this step a decision is made about the number of livestock to place on the pasture this roun - captured in variable livestock

       decide-livestock-placed


     ;; STEP 1C;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
       ;; let livestock feed
       livestock-feed
       ;; check if households have zero livestock and if so, let them die

       ] ;end of first step procedures for EXTENSION agent types
  ]; end of first ask household


 ;STEP 2;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; EXT ONLY
 ; separate additional ask households is needed for agent types in the extension model

 ; print ("")
  ;print ("OBSERVE neighbours")
ask households
      [
         if (member? household-behavioral-type extension-model-behaviors-list)  [
        ;; #EXT once all households have made their decision for current round, and the performance on the pasture calculated (livestock-healthy and destock), agents observe the number of livestock-healthy of their neighbours to decide on future adjustments to their behaviour
        ;; adjustments are made socially by SL1 agent types (to their r-parameter) or by all agents when SL2 global strategy switching behaviour is turned on
        ;; adjustments are made based on the no of livestock healthy (which is still stored in the livestock variable, and not a different one, for compatibility reasons with the original model) as stored at this point in the livestock variable (see livestock value in livestock-feed procedure)
        ;; all adjustments based on social learning (SL1 or SL2) can only be made from second tick onward
            if (SL2-strategy-switch?) OR (household-behavioral-type = "E-RO-SL1")
              [ if ticks >= 2
                  [observe-neighbours]
              ]

         ;    ;; #EXT for extension agents livestock growth would need to happen at the end of the tick
              ;; however, livestock growth implementation is not tested for extension agents and for mixed populations, so this feature is not used in current model version, b rate is set to 0 in extension model, could be developed further
         ;    if extension-model?
         ;    [ if (member? household-behavioral-type extension-model-behaviors-list)
         ;      [
         ;      ;print (word "livestock end of round: " livestock " latest value of destock: " destock)
         ;      set livestock livestock-growth livestock
         ;      ;print (word "livestock placed last time from memory: " array:item household-livestock-placed-memory (ticks - 1))
         ;      ;print (word "memory so far: " household-livestock-placed-memory)
         ;      ]
         ;    ]
         ;
          ]; end of if
       ] ;end of second ask households


  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; patches update reserve biomass values
  ask patches
  [
    ;; reserve biomass growth
    set reserve-biomass ( reserve-biomass-growth green-biomass-init green-biomass reserve-biomass
      reserve-biomass-edible )
  ]

  ;; #EXT update memory of all agents in extension model with new reserve-biomass value
  if extension-model?
  [
    ask households
    [ if (member? household-behavioral-type extension-model-behaviors-list)
      [
      array:set household-reserve-biomass-memory ticks [reserve-biomass] of patch-here
      ]
    ]

  ]



  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; check whether resting is activated and update patches accordingly
;  if resting?
;  [
;    update-resting-patches
;  ]
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; update all plots & view, total livestock sum
  update-color-of-patches

  update-plots

  ifelse ( not extension-model?) ; #added differentiation ext-model
  [set sum-livestock-total sum-livestock-total + sum [ livestock ] of households] ;sum-livestock-total is used to plot SD on livestock graph in interface
  ;#EXT for extension model - sum-livestock-total is sum-livestock-PLACED-total; at the end of the to-go livestock contains livestock-healthy value, hence need to add destock value.
  [set sum-livestock-total sum-livestock-total + sum [livestock + destock] of households]

  set descriptive-norm ifelse-value ( count households > 0 )
  [
    ;; different calculation but yields the same result
    ;; ( count households with [ patch-available-for-grazing? = true ] ) / ( count households )
    mean [ household-resting-behavior ] of households
  ]
  [
    1
  ]
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; HOUSEHOLD MOVEMENT ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to relocate-household

  ; STEP 1:
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; first calculate which patches are available for use based on the DESCRIPTIVE NORM, the own intrinsic preference and the social susceptibility
  ;; #EXT this step is only relevant for original model agents, not for extension agents


  let relevant-patches patches
  ; calculate effective propensity
  set household-effective-propensity household-social-susceptibility * descriptive-norm + ( 1 - household-social-susceptibility ) * household-intrinsic-preference
  ; determine set of patches
  ifelse random-float 1 < household-effective-propensity ;< household-intrinsic-preference
  [
    ; only rested patches are available (within the local neighborhood, as defined by knowledge-radius)
    set relevant-patches patches at-points household-local-neighborhood with [patch-available-for-grazing?]
  ]
  [
    ; all patches are available (within the local neighborhood, as defined by knowledge-radius)
    set relevant-patches patches at-points household-local-neighborhood
  ]
  ; update color of household to reflect it's effective propensity (for TRAD households)
  if ( household-behavioral-type = "TRAD" ) [ set color scale-color violet household-effective-propensity 0 1 ]
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ; STEP 2:
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ifelse (count relevant-patches != 0)
  [
    ;; now that we have determined a set of relevant patches, we need to select one specific patch

    ;; SAT ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    if ( household-behavioral-type = "SAT" )
    [
      ; household is a satisficer, therefore we use satisficing as selection method for the patch
      let patch-found false
      let number-trials 0
      let current-patch patch-here
      let current-best-patch nobody

      ;; update the household satisficing threshold if he currently has less livestock than the global threshold
      set current-household-satisficing-threshold min ( list household-satisficing-threshold livestock )

      ;; first check if the household will be satisfied on his current-patch
      if ( ( [ green-biomass + reserve-biomass-edible ] of current-patch ) / intake >= current-household-satisficing-threshold and local-first? )
      [
        set patch-found true
        update-household-behavior
      ]

      ;; b) by local-first?: sort patches by distance or randomize patch order

      ;; #EXT: why if not local first it should be in order of distance and not random movement?? #TODELETE

      let relevant-patches-ordered ifelse-value local-first? [ sort-on [ distance myself ] relevant-patches ] [ shuffle sort relevant-patches ]
      ;; c) by the number of satisficing-trials: no more than #satisficing-trials patches will be evaluated
      let relevant-patches-ordered-sub sublist relevant-patches-ordered 0 min ( list satisficing-trials length relevant-patches-ordered )

      while [ patch-found = false and number-trials < length relevant-patches-ordered-sub ]
      [
        ;; loop until the end of the patch list is reached or a satisficing patch is found
        set current-patch item number-trials relevant-patches-ordered-sub
        ifelse ( ( [ green-biomass + reserve-biomass-edible ] of current-patch ) / intake >= current-household-satisficing-threshold )
        [
          move-to current-patch
          set patch-found true
          ; check whether selected patch is suitably rested (i.e. available for grazing) and update hh behavior accordingly
          update-household-behavior
        ]
        [
          ;; if the patch doesn't meet the satisficing threshold, save it as current-best-patch if it has higher
          ;; capacity than all previous evaluated patches
          set number-trials number-trials + 1
          ifelse ( current-best-patch = nobody )
          [
            set current-best-patch current-patch
          ]
          [
            set current-best-patch max-one-of ( patch-set current-patch current-best-patch ) [ green-biomass + reserve-biomass-edible ]
          ]
        ]
      ]

      if ( patch-found = false )
      [
        ;; if no satisficing patch is found, move to the best patch that has been found so far
        ifelse current-best-patch != nobody
        [
          move-to current-best-patch
          update-household-behavior
        ]
        [
          set livestock 0
        ;  print (word "just set livestock to 0 here" livestock)
        ]
      ]
    ]
    ;; TRAD / MAX ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    if ( household-behavioral-type = "MAX" or household-behavioral-type = "TRAD" )
    [
      ; households are either TRAD or MAX type, therefore we use maximizing as selection method
      ; move to the patch that provides the highest amount of biomass
      move-to max-one-of relevant-patches [ reserve-biomass-edible + green-biomass ]
      update-household-behavior
    ]
    ;; Random ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    if ( household-behavioral-type = "Random" )
    [
      ;move randomly to one patch
      move-to one-of relevant-patches
      update-household-behavior
    ]

  ] ; end of step 2 ifelse (if set of relevant patches not empty)
  [
    ; no relevant set of patches could be determined, household dies
    set livestock 0
    ;print ( word who ": died, set of relevant patches: " relevant-patches )
    die
  ]

end

 ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
 ;;; EXTENSION AGENT TYPES - DECISION BEHAVIOUR;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
 ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to decide-livestock-placed
    ;;#EXT different procedures for extension model agents follow for each type of agent.
    ;;#EXT for extension model agent types only, the decision on the number of livestock to be placed is made here


    ;<<<<<<<<<<<<<<  E-LBD >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ;; #EXT households do not move
    if (household-behavioral-type = "E-LBD")
    [
     ;print "start"
     ;print ticks
     ;print (word "Household no. " who " with risk " household-risk-att)
      ifelse (ticks >= 2) ; learning can only happen from second tick on
      [

       ; calculate pasture degeneration (negative) or improvement (positive)
        set household-calc-diff ( (array:item household-reserve-biomass-memory (ticks - 1)) - (array:item household-reserve-biomass-memory (ticks - 2)) )
        ;print (word "household-calc-diff: " array:item household-reserve-biomass-memory (ticks - 1) " - " array:item household-reserve-biomass-memory (ticks - 2) ": " household-calc-diff)
        ;calculate rate of change
        let household-calc-diff-ch ( household-calc-diff / (array:item household-reserve-biomass-memory (ticks - 2)) * 100)
        ;print (word "change: " household-calc-diff-ch)


        ; calculate how many livestock should be sold/bought - same rate of change as observed change in reserve biomass
        let livestock-buy-sell (array:item household-livestock-placed-memory (ticks - 2) * household-calc-diff-ch / 100)
        ;initiate second variable to be updated based on risk
        let livestock-buy-sell-new 0

       ; print (word "value in array on position " (ticks - 2) "is " array:item household-livestock-placed-memory (ticks - 2))
        ;print (word "livestock buy sell unrounded " livestock-buy-sell )
        ;print (word "current value of livestock var:" livestock)
        set livestock-buy-sell round-livestock livestock-buy-sell

        ;print (word "livestock-buy-sell: " livestock-buy-sell)


        ; if risk-mode calculate new values for livestock-buy-sell
        if risk-mode?
        [
          ifelse (livestock-buy-sell < 0)
          [
            let x livestock-buy-sell
            let y (x + x * household-risk-att)
            set livestock-buy-sell-new y
          ]
          [ let x livestock-buy-sell
            let y (x + x * household-risk-att * (-1))
           ; let livestock-buy-sell-diff y - livestock-buy-sell
           ; print (word "livestocy buy sell factor to add in calc: " livestock-buy-sell-diff)
            set livestock-buy-sell-new y
          ]
          ;print (word "new BUY SELL adjusted with RISK: " livestock-buy-sell)
         ; NOTE: this livestock-buy-sell-new variable is only used locally in the behaviour of this type, it is not related to the global behaviour of strategy switching (SL2)
          set livestock-buy-sell-new round-livestock livestock-buy-sell-new
          ;print (word "new BUY SELL adjusted with RISK ROUNDED: " livestock-buy-sell)
          ]

        ; update livestock to place on pasture (this round)

        ifelse (destock = 0)
          [
            set livestock array:item household-livestock-placed-memory (ticks - 2) + livestock-buy-sell
            ;print (word "DESTOCK 0: new livestock no to be placed " livestock)
          ]
          [
            set livestock array:item household-livestock-placed-memory (ticks - 2) + min (list livestock-buy-sell (destock * (-1)))
            ;print (word "ELSE: new livestock no to be placed: livestock memory previous round " array:item household-livestock-placed-memory (ticks - 2) "+ min " livestock-buy-sell "AND " destock " destock*(-1) is " livestock )
          ]

        if risk-mode?
        [ ;print (word "livestock rec is: " livestock)
          ; in risk mode livestock is calculated as a deviation from the livestock that would have been calculated without risk, hence no ELSE is necessary here.
          ; make sure new livestock value cannot take negative values (when livestock init is very big)
          set livestock max (list (livestock + livestock-buy-sell-new - livestock-buy-sell) 0)
          ;print (word "NEW LIVESTOCK ADDED with risk is: livestock rec + b/s-new - b/s: rec + " livestock-buy-sell-new " - " livestock-buy-sell " = " livestock)
        ]


         array:set household-livestock-placed-memory (ticks - 1) livestock
         ;print (word "added to memory of livestock on:" (ticks - 1) " this value: " livestock )

          ]

      [ ;if ticks < 2, i.e. ticks = 1
          set livestock array:item household-livestock-placed-memory 0
         ;print (word "Just reset the livestock variable back to the original value from init: " livestock)
      ]

        ]
    ; end of E-LBD profile


    ;<<<<<<<<<<<<<< E-RO & E-RO-SL1  >>>>>>>>>>>>>>>>>>>>>>>>>>>>- extension model, risk only profile, destock as in baseline model (MAX) ;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ;; #EXT decision on no of livestock to place on the pasture this tick is taken here
    ;; #EXT households do not move
    ;; the decision behaviour for these profiles does not depend on whether risk-mode is on or not. If risk-mode were off, they would behave just like MAX type in terms of destocking, but with no movement.
    ;; these types must therefore have risk. Risk-mode switch is only for LBD agent type.

    if (household-behavioral-type = "E-RO") OR (household-behavioral-type = "E-RO-SL1")
    [

      ;print "DECISION E-RO/E-RO-SL1"
      ;print (word "Household no. " who " with risk " household-risk-att " and type: " household-behavioral-type)

      ifelse (ticks = 1)
      [ set livestock array:item household-livestock-placed-memory 0  ;this is probably redundant, to check and maybe delete
        ;print (word "Just make sure that the livestock variable contains the original value from init: " livestock)
      ]
      [ ; if ticks >=2
        ; adjustment with risk attitude will only happen from second round, first need observation of no of livestock gone hungry (captured in destock variable)
        ; observe how many livestock have gone hungry (destock variable value), then adjust destock value according to r param; new livestock to be placed will be (livestock no from last round) - (adjusted destock)

        set livestock risk-adjust-livestock livestock ; calculates adjusted livestock (based on r-param value) to be placed on the pasture

        ; add livestock no placed on pasture to memory
        array:set household-livestock-placed-memory (ticks - 1) livestock
        ;print (word "added to memory of livestock on:" (ticks - 1) " this value: " livestock )
      ]
    ]
    ; end of E-RO and E-RO-SL1 decision profiles

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; LIVESTOCK FEEDING ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to livestock-feed
  ;; calculate livestock fodder needed
  let fodder-needed intake * livestock
  ;print ("")
  ;print ("livestock feed")
  ;print (word "fodder-needed: " fodder-needed " for " livestock  " livestock");#EXT

  ;; based on the caluclated fodder, there can be three cases:
  ;; case 1: destocking: yes, green-biomass: all consumed, edible reserve biomass: all consumed
  ifelse fodder-needed > (reserve-biomass-edible + green-biomass)
  [
    ;; not enough biomass available, all biomass will be consumed and part of the livestock will be destocked
    let current-destock ceiling ( ( fodder-needed - (reserve-biomass-edible + green-biomass) ) / intake )
    ;print (word "reserve-biomass-edible: " reserve-biomass-edible " green biomass: " green-biomass " current destock: " current-destock) ;#EXT

    set livestock ( livestock - current-destock )
   ; EXT in extension model the value that the livestock variable takes here actually corresponds to livestock-healthy, i.e. (livestock-placed this round) - (current destock) -> this value is the one used when observing neighbour performance
    ;print (word "livestock variable after destocking bc not enough green+res biomass " livestock) ;#EXT
    ;log output
    ;if ( livestock <= 0 )
    ;[
    ;  print ( word who ": livestock set to zero while feeding, will die now. livestock = " livestock )
    ;]

    ifelse extension-model?

    [set destock current-destock ; #EXT for extension types this represents
     ;print (word "new destock value = current-destock: " destock) ;#EXT
    ]

    [ ; this is the destock for the original model agents, in #EXT it is replaced by the one above in the learning model
      set destock destock + current-destock
    ]

      ;; no green biomass and no edible reserve biomass is left
    set green-biomass 0
    set reserve-biomass-edible 0
  ]
  [
    ;; case 2: destocking: no, green-biomass: all consumed, edible reserve biomass: partially consumed
    ifelse fodder-needed > green-biomass
    [
      ;; all green biomass is consumed and some part of edible reserve biomass as well,
      ;; but livestock does not need to be destocked
      set reserve-biomass-edible ( reserve-biomass-edible - ( fodder-needed - green-biomass ) )
      set green-biomass 0
      if extension-model?
      [
        set destock 0 ;#EXT
      ]
    ]
    [
      ;; case 3: destocking: no, green-biomass: partially consumed, edible reserve biomass: not consumed
      set green-biomass ( green-biomass - fodder-needed )
      if extension-model?
      [
      set destock 0 ; #EXT this is because of the sequence of procedures in to go: first movement (decision for learning agents), then destock (in livestock-feed proc), but the last value of the destock variable is used in the decision of the ext agent types
      ]
    ]
  ]

  ;; keep track of total livestock and destock over time
  ;; livestock-healthy shown here
  set livestock-healthy-total livestock-healthy-total + livestock
  set destock-total destock-total + destock

  ;; as long as the livestock is > 0 (i.e. not all animals have been destocked), the patch has been grazed in this year
  ifelse livestock > 0
  [
    set is-being-grazed? true
  ]
  [
    set is-being-grazed? false
  ]


end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; OTHER PROCEDURES FROM ORIGINAL MODEL -- not used in extension model -- moved in separate procedures for improved readability

;; RAIN and RESTING ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; setting rain conditions for each pasture and also update patches resting status according to separate procedure (if resting policy is enabled)
;; this procedure is called first every tick in the to-go procedure
to rain-and-resting
if ( global-rain? )
    [
      ifelse ( use-rain-from-file? )
      [
        set global-rain item ticks global-rainfall-list
      ]
      [
        ;; calculate one global rainfall value if option global-rain? is checked
        set global-rain rainfall
      ]
    ]

    if resting?
    [
      update-resting-patches
    ]

end

;; GREEN BIOMASS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; procedure calculating the new available green biomass - as a function of remaining green-biomass after grazing, rain, and remaining reserve-biomass after grazing
;; also calculates reserve-biomass-edible in this current round
;; called second in to-go, after rain and resting conditions, but before the households get to make their decisions (household movement)
to update-green-biomass-new-tick
    ;; #EXT in extension-model skip following command block for rainfall for every tick, rain stays constant at the fixed value established in setup procedure
    if not extension-model?
    [
      ;; rainfall
      ifelse ( global-rain? )
      [
        set rain global-rain
      ]
      [
        set rain rainfall
      ]
    ]
    ;; green biomass growth
    set green-biomass-init ( green-biomass-growth green-biomass rain reserve-biomass )
    set green-biomass green-biomass-init
    ;; new year, so patches have not been grazed yet
    set is-being-grazed? false
    ; calculate the amount of reserve biomass that can be consumed in this tick
    set reserve-biomass-edible reserve-biomass * gr2

    if ( any? households with [ household-behavioral-type = "TRAD" ] )
    [
      ;set descriptive-norm mean [ household-intrinsic-preference ] of households
    ]
end


to destock-SAT
  ;; when strategy "satisficing-2" is chosen, households will stock no more than their satisficing threshold
  ;; check whether herd size > satisficing-threshold and destock animals, if necessary
  let current-destock-sat max ( list 0 (livestock - household-satisficing-threshold ) )
  set livestock ( livestock - current-destock-sat )
  set destock destock + current-destock-sat
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; POLICY INTERVENTIONS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; RESTING ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
to update-resting-patches
  ;; check all rested patches in this year
  ask patches with [ reserve-biomass < resting-threshold * Rmax ]
  [
    set is-rested? true
    set patch-available-for-grazing? false
    set plabel "R"
    set plabel-color red
  ]
  ask patches with [ reserve-biomass >= resting-threshold * Rmax ]
  [
    set is-rested? false
    set patch-available-for-grazing? true
    set plabel "G"
    set plabel-color white
  ]
end

to update-household-behavior
  ; check whether selected patch is suitably rested (i.e. available for grazing) and update hh behavior accordingly
  ifelse ( patch-available-for-grazing? )
  [
    set household-resting-behavior 1
  ]
  [
    set household-resting-behavior 0
  ]
end
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; OTHER EXTENSION PROCEDURES / REPORTERS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;; ROUND livestock no ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;#EXT function to be called for rounding livestock numbers when adjustment results in decimals

to-report round-livestock [livestock-x]
        ; round to the nearest integer up, if value is negative, and to the nearest integer down if positive (because unit is livestock)
        ifelse (livestock-x < 0)
        [
          set livestock-x ceiling livestock-x
        ]
        [
          set livestock-x floor livestock-x
        ]
  report livestock-x
end


;; Risk-based ADJUSTMENT of livestock no ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; #EXT function to be called to adjust livestock no with risk only profile (E-RO) or as part of social learning behavior type (SL1)

to-report risk-adjust-livestock [livestock-x]
   ; calculate pasture degeneration (negative) or improvement (positive)
   ; print (word "last destock value was: " destock)
   ; destock value contains the no of livestock that were hungry in previous round and it has been last updated in previous tick in the "livestock-feed" procedure

     let destock-adj round-livestock (destock + destock * household-risk-att)
   ;  print (word "adjusted destock value rounded is: " destock-adj)

     ; update livestock to place on pasture (this round)
     ; make sure no negative values are allowed for livestock to be placed
     set livestock-x max (list (array:item household-livestock-placed-memory (ticks - 2) - destock-adj) 0)
   ;  print (word "new livestock to be placed this round is: " livestock-x)

  report livestock-x
end



;; OBSERVING NEIGHBOURS' performance - for SL1 and SL2 strategy switching ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; #EXT procedure -- currently performance is measured in terms of "livestock healthy", it could also be based on ecological performance in future versions
to observe-neighbours
    set new-household-behavioral-type "none"
    set new-household-risk-att "none"
    set new-color "none"

    ;identify neighbouring patches within knowledge radius
     let neighbor-patches patches at-points household-local-neighborhood

     ;print (word "household: "who " - this patch is: " patch-here "neighbor-patches are: " neighbor-patches)

     ;identify neighbouring households on neighbour patches
     let neighbor-households (households-on neighbor-patches)
     ;print (word "neighbor-households are: " neighbor-households)

     ; check if any neighbours found
     ifelse (any? neighbor-households)
     [
        ;check if there are any more succesful agents than self from whom to learn
        ;the basis for comparison is the value of the [livestock] variable which at this moment stores (for ext agents) the (livestock-placed) - (destock) --> so the no of livestock healthy
        let successful-neighbor max-one-of neighbor-households [livestock]

        ;check if succesful neighbor is more succesful than self, if yes, remember their behaviour type, otherwise do the same thing as before (tick 2)

          ifelse ([livestock] of successful-neighbor > livestock)
               [
               ; if the reason the observe-neighbours procedure is running is the SL2 global switching behaviour turned on, then note the household-behavioral-type of the successful neighbour:
              ;print (word "**********" [who] of successful-neighbor " has more livestock than oneself, i.e.: " [livestock] of successful-neighbor " > " livestock)

              if SL2-strategy-switch?
                  [ifelse ([household-behavioral-type] of successful-neighbor != household-behavioral-type)
                       [set new-household-behavioral-type [household-behavioral-type] of successful-neighbor
                        set new-color [color] of successful-neighbor
                        ;print (word who ": behavior will be adopted from successful neighbor " [who] of successful-neighbor ". New value next round is " [household-behavioral-type] of successful-neighbor)
                       ]
                       [;print (word "behavior of successful neighbor same as own, not retained for adoption")
                       ]
                  ]

               ; if the household calling this procedure is SL1 type, then (also) note the risk attitude of the successful-neighbor.
               if (household-behavioral-type = "E-RO-SL1")
                   [ifelse ([household-risk-att] of successful-neighbor != household-risk-att)
                      [ set new-household-risk-att [household-risk-att] of successful-neighbor
                    ;print (word who ": r-param " [household-risk-att] of successful-neighbor " will be adopted from successful neighbor " [who] of successful-neighbor ". Current r-param is: " household-risk-att)
                      ]
                      [ ;print (word "r-param of successful neighbor same as own, not retained for adoption")
                      ]
                   ]
                ]
                ;else, most successful neighbor has less livestock-healthy than self
                [ ;print (word "no neighbour with more livestock")
                ]
    ]
    [
     ;print (word who " has no neighbours" )
    ]
end


;; STRATEGY SWITCHING / SL2 ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; #EXT function for strategy-switching
; when SL2 global strategy switching behaviour is active (on), then this procedure updates household strategy by changing agent type with the successful agent type recorded when observing neighbours
to update-household-strategy
 ; first observation of neighbours happens in tick 2, so updates can only take place from tick 3 onward.

  ;print (word "UPDATING strategy")
  ;print (word who ": new round, old beh type is: " household-behavioral-type)
  ifelse  (ticks >= 3) AND (new-household-behavioral-type != "none")
      [
      set household-behavioral-type new-household-behavioral-type
      set color new-color
      ;set color yellow
      ;wait 0.3
      set household-SL2? true
      ;print (word who ": new behavior is active, i.e. " household-behavioral-type)
      ]
      [ ifelse  (ticks >= 3)
          [
           ;print (word who ": no better strategy identified, continuing with old type")
          ]
          [;print (word who ": first/second tick, not applicable")
          ]
      ]
end

;; SL1 / UPDATING RISK ATTITUDE
;; #EXT function called by SL1 agents only - update r parameter value at the beginning of the tick with the values observed in successful agents (during observe neighbours procedure)
to update-r-param
  ; first observation of neighbours happens in tick 2, so updates can only take place from tick 3 onward.
     ifelse (ticks >= 3) AND (new-household-risk-att != "none")
      [
        ;print (word "UPDATING r-param")
      set household-risk-att new-household-risk-att

      ifelse household-risk-att >= 0
             [set shape "triangle" ]
             [set shape "triangle-down" ]

      set color red
      wait 0.3
      set color lime
      ; set variable to show that social learning has happened
      set household-SL? true
       ;print (word who ": new r-parameter value is active, i.e. " household-risk-att)
      ]
      [ifelse (ticks >= 3)
       [;print (word who ": no better r-parameter identified, continuing with old value")
       ]
       [;print (word who ": first/second tick, not applicable")
       ]
  ]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
@#$#@#$#@
GRAPHICS-WINDOW
405
10
682
288
-1
-1
26.9
1
10
1
1
1
0
0
0
1
-4
5
-4
5
1
1
1
ticks
30.0

BUTTON
460
320
545
353
NIL
setup
NIL
1
T
OBSERVER
NIL
P
NIL
NIL
1

BUTTON
555
375
640
408
NIL
go
T
1
T
OBSERVER
NIL
G
NIL
NIL
1

PLOT
970
190
1240
310
green biomass [average +/- sd]
timestep
biomass
0.0
10.0
0.0
70000.0
true
false
"" "set-plot-x-range 0 timesteps\nset-plot-y-range 0 ceiling max ( list plot-y-max ( mean [ green-biomass ] of patches + standard-deviation [ green-biomass ] of patches ) )"
PENS
"green-biomass" 1.0 0 -13840069 true "" "plotxy ticks mean [ green-biomass ] of patches"
"pen-2" 1.0 0 -5509967 true "" "plotxy ticks ( mean [ green-biomass ] of patches - downside-sd [ green-biomass ] of patches )"
"pen-3" 1.0 0 -5509967 true "" "plotxy ticks ( mean [ green-biomass ] of patches + upside-sd [ green-biomass ] of patches )"

PLOT
970
315
1240
435
rainfall [average +/- sd]
timestep
rainfall
0.0
10.0
0.0
10.0
true
false
"" "set-plot-x-range 0 timesteps\nset-plot-y-range 0 ceiling max ( list plot-y-max ( mean [ rain ] of patches + standard-deviation [ rain ] of patches ) )"
PENS
"mean rainfall" 1.0 0 -13345367 true "" "plotxy ticks mean [ rain ] of patches"
"rainfall-sd" 1.0 0 -5325092 true "" "plotxy ticks ( mean [ rain ] of patches - standard-deviation [ rain ] of patches )"
"rainfall+sd" 1.0 0 -5325092 true "" "plotxy ticks ( mean [ rain ] of patches + standard-deviation [ rain ] of patches )"

BUTTON
460
375
545
408
step
go
NIL
1
T
OBSERVER
NIL
S
NIL
NIL
1

SLIDER
130
70
285
103
number-households
number-households
1
100
60.0
1
1
NIL
HORIZONTAL

PLOT
690
10
965
185
livestock (EXT: healthy) [average +/- sd]
timestep
livestock
0.0
10.0
0.0
10.0
false
false
"" "set-plot-x-range 0 timesteps\nset-plot-y-range 0 ceiling max ( list plot-y-max ( mean [ livestock ] of households + upside-sd [ livestock ] of households ) )"
PENS
"livestock" 1.0 0 -2674135 true "" "if count households > 0 [ plotxy ticks mean [ livestock ] of households ]"
"livestock-sd" 1.0 0 -1604481 false "" "if count households > 1 [ plotxy ticks ( mean [ livestock ] of households - downside-sd [ livestock ] of households ) ]"
"livestock+sd" 1.0 0 -1604481 false "" "if count households > 1 [ plotxy ticks ( mean [ livestock ] of households + upside-sd [ livestock ] of households ) ]"

PLOT
970
10
1240
185
reserve biomass [average +/- sd]
timestep
biomass
0.0
10.0
0.0
150000.0
true
false
"" "set-plot-x-range 0 timesteps\nset-plot-y-range 0 ceiling max ( list plot-y-max ( mean [ reserve-biomass ] of patches + standard-deviation [ reserve-biomass ] of patches ) )"
PENS
"default" 1.0 0 -14835848 true "" "plotxy ticks mean [ reserve-biomass ] of patches"
"pen-1" 1.0 0 -5908279 true "" "plotxy ticks ( mean [ reserve-biomass ] of patches - downside-sd [ reserve-biomass ] of patches )"
"pen-2" 1.0 0 -5908279 true "" "plotxy ticks ( mean [ reserve-biomass ] of patches + upside-sd [ reserve-biomass ] of patches )"

INPUTBOX
130
195
205
255
w
0.8
1
0
Number

PLOT
690
190
965
310
current (EXT: healthy) livestock [each household]
household
livestock
0.0
100.0
0.0
100.0
false
false
"" ""
PENS
"livestock-current" 1.0 1 -2674135 true "" "plot-livestock-per-household"

INPUTBOX
210
195
290
255
gr1
0.5
1
0
Number

INPUTBOX
295
195
375
255
gr2
0.1
1
0
Number

INPUTBOX
290
80
375
140
b
0.0
1
0
Number

TEXTBOX
130
150
310
168
Vegetation parameters
13
0.0
1

TEXTBOX
130
45
355
63
Household & livestock parameters
13
0.0
1

CHOOSER
5
290
165
335
behavioral-type
behavioral-type
"MAX" "SAT" "TRAD" "Random" "E-RO" "E-LBD" "E-RO-SL1"
4

PLOT
690
315
965
435
(ORIG: surviving) households [ livestock > 0 ]
timestep
household count
0.0
10.0
0.0
10.0
true
false
"" "set-plot-x-range 0 timesteps\n"
PENS
"households" 1.0 0 -16777216 true "" "plot count households with [ livestock > 0 ]"
"pen-1" 1.0 0 -2674135 true "" "plot count households with [ livestock > 0 and household-behavioral-type = \"MAX\" ]"
"pen-2" 1.0 0 -8630108 true "" "plot count households with [ livestock > 0 and household-behavioral-type = \"TRAD\" ]"
"pen-3" 1.0 0 -13345367 true "" "plot count households with [ livestock > 0 and household-behavioral-type = \"SAT\" ]"
"pen-4" 1.0 0 -7500403 true "" "plot count households with [ livestock > 0 and household-behavioral-type = \"E-RO\" ]"
"pen-5" 1.0 0 -11221820 true "" "plot count households with [ livestock > 0 and household-behavioral-type = \"E-LBD\" ]"
"pen-6" 1.0 0 -13840069 true "" "plot count households with [ livestock > 0 and household-behavioral-type = \"E-RO-SL1\" ]"

TEXTBOX
290
65
370
83
sheep birth rate
11
0.0
1

TEXTBOX
130
165
210
195
biomass growth rate
11
0.0
1

TEXTBOX
210
165
295
195
green biomass grazing pressure
11
0.0
1

TEXTBOX
295
165
380
195
reserve biomass grazing pressure
11
0.0
1

SWITCH
5
160
120
193
global-rain?
global-rain?
0
1
-1000

TEXTBOX
5
140
155
158
Rainfall parameters
13
0.0
1

INPUTBOX
5
195
65
255
rain-mean
200.0
1
0
Number

INPUTBOX
70
195
120
255
rain-std
100.0
1
0
Number

TEXTBOX
5
15
285
40
RAGE RAngeland Grazing ModEl
18
0.0
1

SLIDER
5
410
165
443
satisficing-threshold
satisficing-threshold
0
100
0.0
1
1
NIL
HORIZONTAL

SLIDER
175
525
390
558
satisficing-trials
satisficing-trials
0
100
21.0
1
1
NIL
HORIZONTAL

SLIDER
5
525
165
558
knowledge-radius
knowledge-radius
0
5
5.0
1
1
patches
HORIZONTAL

PLOT
805
470
965
590
Lorenz Curve
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"lorenz" 1.0 0 -16777216 true "" "plot-lorenz"
"pen-1" 1.0 0 -7500403 true "" "plotxy 0 0\nplotxy number-households 1"

MONITOR
690
520
800
565
Total livestock
mean-total-livestock
2
1
11

MONITOR
690
470
800
515
Current mean livestock
mean-livestock
2
1
11

SWITCH
130
105
285
138
start-on-homepatch?
start-on-homepatch?
0
1
-1000

TEXTBOX
5
490
155
508
Knowledge of pastures
13
0.0
1

TEXTBOX
5
265
205
283
Behavioral types
16
0.0
1

MONITOR
1135
470
1215
515
Gini coef
gini-coef-surviving
3
1
11

SWITCH
400
525
530
558
resting?
resting?
1
1
-1000

TEXTBOX
400
500
530
520
Pasture resting
16
0.0
1

INPUTBOX
5
80
105
140
timesteps
100.0
1
0
Number

TEXTBOX
10
45
130
86
Simulation length 
13
0.0
1

SLIDER
300
375
420
408
number-random
number-random
0
number-households - number-E-RO-SL1 - number-E-RO - number-E-LBD - number-MAX - number-SAT - number-TRAD
0.0
1
1
NIL
HORIZONTAL

SLIDER
175
340
295
373
number-MAX
number-MAX
0
number-households - number-E-RO-SL1 - number-E-RO - number-E-LBD - number-random - number-SAT - number-TRAD
0.0
1
1
NIL
HORIZONTAL

SLIDER
175
375
295
408
number-SAT
number-SAT
0
number-households - number-E-RO-SL1 - number-E-RO - number-E-LBD - number-MAX - number-random - number-TRAD
0.0
1
1
NIL
HORIZONTAL

SLIDER
300
340
420
373
number-TRAD
number-TRAD
0
number-households - number-E-RO-SL1 - number-E-RO - number-E-LBD - number-MAX - number-SAT - number-random
0.0
1
1
NIL
HORIZONTAL

PLOT
970
470
1130
590
Descriptive norm
NIL
NIL
0.0
10.0
0.0
1.0
true
false
"" "set-plot-x-range 0 timesteps\nset-plot-y-range 0 1"
PENS
"default" 1.0 0 -16777216 true "" "plot descriptive-norm"

SLIDER
535
525
680
558
resting-threshold
resting-threshold
0
1
0.1
0.01
1
NIL
HORIZONTAL

SWITCH
175
305
420
338
homog-behav-types?
homog-behav-types?
1
1
-1000

SLIDER
5
340
165
373
intrinsic-preference
intrinsic-preference
0
1
0.0
0.01
1
NIL
HORIZONTAL

SLIDER
5
375
165
408
social-susceptibility
social-susceptibility
0
1
0.0
0.01
1
NIL
HORIZONTAL

TEXTBOX
175
275
390
301
For mixed behavioral types, set to \"off\" and select numbers below
11
0.0
1

TEXTBOX
5
510
155
528
relevant for all types:
11
0.0
1

TEXTBOX
175
510
325
528
relevant only for SAT type:
11
0.0
1

SWITCH
10
610
162
643
extension-model?
extension-model?
0
1
-1000

INPUTBOX
170
610
270
670
livestock-init
90.0
1
0
Number

PLOT
1255
35
1620
210
Mean reserve biomass by type of agent
NIL
NIL
0.0
100.0
0.0
10.0
true
true
"" ""
PENS
"E-RO" 1.0 0 -7500403 true "" "if (count households with [household-behavioral-type = \"E-RO\"]) > 0 [plotxy ticks mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = \"E-RO\"] ] ]"
"E-LBD" 1.0 0 -11221820 true "" "if (count households with [household-behavioral-type = \"E-LBD\"]) > 0 [plotxy ticks mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = \"E-LBD\"] ]]"
"E-RO-SL1y" 1.0 0 -2064490 true "" "if (count households with [household-SL? = TRUE]) > 0 [plotxy ticks mean [reserve-biomass] of patches with [ any? households-here with [(household-SL? = TRUE)] ] ]"
"E-RO-SL1all" 1.0 0 -13840069 true "" "if (count households with [household-behavioral-type = \"E-RO-SL1\"]) > 0 [plotxy ticks mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = \"E-RO-SL1\")]]]"
"SL2y" 1.0 0 -612749 true "" "if (count households with [household-SL2? = TRUE]) > 0 [plotxy ticks mean [reserve-biomass] of patches with [ any? households-here with [(household-SL2? = TRUE)] ]]"

PLOT
1255
215
1620
375
Livestock placed of first 5 HH
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"mean-lv-P" 1.0 0 -16777216 true "" "plot mean [livestock + destock] of households"
"HH0" 1.0 0 -2674135 true "" "plot [livestock + destock] of household 0"
"HH1" 1.0 0 -7500403 true "" "plot [livestock + destock] of household 1"
"HH2" 1.0 0 -955883 true "" "plot [livestock + destock] of household 2"
"HH3" 1.0 0 -6459832 true "" "plot [livestock + destock] of household 3"
"HH4" 1.0 0 -1184463 true "" "plot [livestock + destock] of household 4"

SLIDER
175
430
295
463
number-E-RO
number-E-RO
0
number-households - number-E-RO-SL1 - number-E-LBD - number-MAX - number-random - number-TRAD - number-SAT
20.0
1
1
NIL
HORIZONTAL

SLIDER
300
430
420
463
number-E-LBD
number-E-LBD
0
number-households - number-E-RO-SL1 - number-E-RO - number-MAX - number-random - number-TRAD - number-SAT
20.0
1
1
NIL
HORIZONTAL

PLOT
1255
380
1620
540
Average livestock placed per agent type
NIL
NIL
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"E-RO" 1.0 0 -7500403 true "" "if (count households with [household-behavioral-type = \"E-RO\"]) > 0 [ plotxy ticks mean [ livestock + destock] of households with [ household-behavioral-type = \"E-RO\" ] ]\n"
"E-LBD" 1.0 0 -11221820 true "" "if (count households with [household-behavioral-type = \"E-LBD\"]) > 0  [ plotxy ticks mean [ livestock ] of households with [ household-behavioral-type = \"E-LBD\" ] ]\n"
"E-RO-SL1y" 1.0 0 -2064490 true "" "if (count households with [household-SL? = TRUE]) > 0 [ plotxy ticks mean [ livestock ] of households with [(household-SL? = TRUE) ] ]\n"
"E-RO-SL1all" 1.0 0 -13840069 true "" "if (count households with [household-behavioral-type = \"E-RO-SL1\"]) > 0  [ plotxy ticks mean [ livestock ] of households with [ (household-behavioral-type = \"E-RO-SL1\") ] ]"
"SL2y" 1.0 0 -612749 true "" "if (count households with [household-SL2? = TRUE]) > 0 [ plotxy ticks mean [ livestock ] of households with [ (household-SL2? = TRUE) ] ]\n"

SLIDER
175
465
295
498
number-E-RO-SL1
number-E-RO-SL1
0
number-households - number-E-LBD - number-E-RO - number-MAX - number-random - number-TRAD - number-SAT
20.0
1
1
NIL
HORIZONTAL

MONITOR
1255
550
1400
595
E-RO mean total livestock P
mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type = \"E-RO\"]
0
1
11

MONITOR
1255
600
1400
645
E-LBD mean total livestock P
mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type = \"E-LBD\"]
0
1
11

MONITOR
1403
550
1523
595
E-RO mean destock
mean [destock-total] of households with [household-behavioral-type = \"E-RO\"]
0
1
11

MONITOR
1403
600
1523
645
E-LBD mean destock
mean [destock-total] of households with [household-behavioral-type = \"E-LBD\"]
0
1
11

SWITCH
280
635
435
668
SL2-strategy-switch?
SL2-strategy-switch?
1
1
-1000

MONITOR
1403
650
1523
695
E-RO-SL1 mean destock
mean [destock-total] of households with [household-behavioral-type = \"E-RO-SL1\"]
0
1
11

MONITOR
1253
650
1398
695
E-RO-SL1 mean total livestock P
mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type = \"E-RO-SL1\"]
0
1
11

TEXTBOX
15
580
295
611
Extension Model - Learning agents
16
0.0
1

TEXTBOX
1260
10
1530
46
Extension Model Outputs
16
0.0
1

TEXTBOX
175
415
360
441
relevant only for extension model:
11
0.0
1

TEXTBOX
285
615
530
641
Switch for global behaviour: strategy switching
11
0.0
1

TEXTBOX
10
65
160
83
(extension: max 500)
11
0.0
1

PLOT
1530
550
1805
680
current livestock-placed [each household]
household
livestock
0.0
100.0
0.0
100.0
true
false
"" ""
PENS
"total-livestock-placed-current" 1.0 1 -2674135 true "" "plot-total-livestock-placed-per-household "

MONITOR
1630
495
1712
540
Gini coef EXT
gini-coef-total-livestock-healthy
3
1
11

SLIDER
5
830
160
863
risk-att-init
risk-att-init
-0.95
0.95
0.5
0.05
1
NIL
HORIZONTAL

SWITCH
10
725
160
758
risk-mode?
risk-mode?
0
1
-1000

TEXTBOX
10
710
160
728
relevant only for E-LBD types:
11
0.0
1

TEXTBOX
10
770
160
826
INACTIVE (slider only to be used for BehaviourSpace experiment when activated from the code)
11
0.0
1

TEXTBOX
10
685
160
703
For Experiments ONLY
14
0.0
1

@#$#@#$#@
# RAGE RAngeland Grazing ModEl - EXTENSION

## WHAT IS IT?

This is an extension of the original RAGE model where we add learning capabilities to agents, specifically learning-by-doing and social learning (two processes central to adaptive (co-)management). 

The original model can be found here:
https://www.comses.net/codebases/5721/releases/1.0.0/

 
The RAGE model is a *multi-agent simulation model* that captures *feedbacks between pastures, livestock and household livelihood in a common property grazing system*. It implements three stylized *household behavioral types* (traditional *TRAD*, maximizer *MAX* and satisficer *SAT*) which are grounded in social theory and reflect empirical observations. These types can be compared regarding their long-term social-ecological consequences. The types differ in their preferences for livestock, how they value social norms concerning pasture resting and how they are influenced by the behavior of others. 

Besides the evaluation of the behavioral types, the model allows to adjust a range of ecological and climatic parameters, such as rainfall average and variability, vegetation growth rates or livestock reproduction rate. The model can be evaluated across a range of social, ecological and economic outcome variables, such as average herd size, pasture biomass condition or surviving number of households.

**The extension module** is applied to smallholder farmers' decision-making - here, a pasture (patch) is the private property of the household (agent) placed on it and there is no movement of the households. Households observe the state of the pasture and their neighrbours to make decisions on how many livestock to place on their pasture every year. Three new behavioural types are created (which cannot be combined with the original ones): E-RO (baseline behaviour), E-LBD (learning-by-doing) and E-RO-SL1 (social learning). Similarly to the original model, these three types can be compared regarding long-term social-ecological performance. In addition, a global strategy switching option (corresponding to double-loop learning) allows users to study how behavioural strategies diffuse in a heterogeneous population of learning and non-learning agents. 

An important modification of the original model is that extension agents are heterogeneous in how they deal with uncertainty. This is represented by an agent property, called the r-parameter (household-risk-att in the code). The r-parameter is catch-all for various factors that form an agent's disposition to act in a certain way, such as: uncertainty in the sensing (partial observability of the resource system), noise in the information received, or an inherent characteristic of the agent, for instance, their risk attitude. 


### A detailed description of the model and its processes can be found in the accompanying ODD+D protocol

## HOW IT WORKS

### Yearly sequence of processes

The model consists of two main sub-models:  _**Vegetation**_, and _**Household decision-making**_. Those components are linked by the following **yearly sequence**, which processes each **tick**:

1. Green biomass is updated (with fixed rain, for comparability purposes)
2. Households with E-RO-SL1 update their r-parameter
3. Households update their behavioural strategy (if strategy switching enabled)
4. Households make decisions on how many livestock to place on their pasture in the next round
5. Livestock feeds on biomass
6. Households observe the performance of their neighbours
7. Reserve biomass is updated
8. Households observe and remember reserve biomass

### Model parameters 

Extension model parameters: Standard value / range

- extension-model?: true
- risk-mode?: true
- number-households: 60   
- timesteps: 100
- b: 0 (reproduction not used in extension model)
- knowledge-radius: 1
- homog-behav-types?: true
- number-E-RO, number-E-LBD, number-E-RO-SL1: 20, 20, 20
- rain: 282.71
- SL2-strategy-switch?: false
- behavioral-type: E-RO


Original model parameters used by the extension model:

- w: 0.8
- gr1: 0.5
- gr2: 0.1
- rue: 0.002
- mg: 0.1
- mr: 0.05
- R0part: 0.6
- lambda: 0.5
- Rmax: 0.5
- intake: 640


## Things to try

### Influence of demographic settings and of herd size

Varying the initial *number-households* and the initial *livestock-init* in the system.

### Impact of various learning strategies

Comparing the performance of homogeneous groups of agents (*homog-behav-types?* true) of each type: non-learning agents (*behavioral-type* E-RO), learning-by-doing (*behavioral-type* E-LBD), and social learning single-loop (*behavioral-type* E-RO-SL1).

### Impact of social learning douple-loop (strategy switching)

Comparing the results of heterogeneous groups of households (*homog-behav-types?* false) with strategy switching enabled vs. disabled (*SL2-strategy-switch?* true or false).

### Diffusion of learning behaviors // "Battle" of learning types

Observe visually the evolution of numbers of agents of each type (corresponding to different colors) in a heterogeneous population (*homog-behav-types?* false) of agents.

## REFERENCES 

### Original

Dressler, G., Groeneveld, J., Buchmann, C.M., Guo, C., Hase, N., Thober, J., Frank, K. and Müller, B. (2018): _**Implications of behavioral change for the resilience of pastoral systems – lessons from an agent-based model**_, Ecological Complexity.

Dressler, G., Müller, B. and Frank, K. (2012): _**Mobility – a panacea for pastoralism? An ecological-economic modelling approach.**_, *Proceedings of the iEMSs Fifth Biennial Meeting: International Congress on Environmental Modelling and Software (iEMSs 2012)*. International Environmental Modelling and Software Society, Leipzig, Germany, July 2012.

Martin, R., Müller, B., Linstädter, A. and Frank, K. (2014): _**How much climate change can pastoral livelihoods tolerate? Modelling rangeland use and evaluating risk.**_, *Global Environmental Change*, 24, 183-192.

Müller, B., Frank, K. and Wissel, C. (2007): _**Relevance of rest periods in non-equilibirum rangeland systems - a modelling analysis**_, *Agricultural Systems*, 92, 295–317.

Schulze, J. (2011): _**Risk Measures and Decision Criteria for the Management of Natural Resources Under Uncertainty - Application to an Ecological-Economic Grazing Model**_, *Master Thesis*, Helmholtz Centre for Environmental Research & Ernst- Moritz-Arndt-University of Greifswald.
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

arrow-down
true
0
Polygon -7500403 true true 150 300 0 150 105 150 105 7 195 7 195 150 300 150

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

house2
false
0
Rectangle -7500403 true true 15 120 285 285
Polygon -7500403 true true 15 120 150 15 285 120

house3
false
0
Rectangle -7500403 true true 15 75 195 180
Polygon -7500403 true true 15 75 105 15 195 75
Polygon -7500403 true true 120 285 120 255 105 240 105 210 120 195 195 195 210 210 225 195 240 195 270 210 270 225 255 240 240 240 210 255 210 285 195 285 195 255 135 255 135 285
Polygon -7500403 true true 105 210 90 270 105 240
Polygon -7500403 true true 195 285 180 210 225 210 210 285
Polygon -7500403 true true 135 285 150 210 120 210 105 240
Polygon -7500403 true true 240 195 210 180 225 195
Polygon -7500403 true true 240 195 255 195 255 210

house4
false
0
Rectangle -7500403 true true 15 75 165 195
Polygon -7500403 true true 15 75 90 15 165 75
Circle -7500403 true true 208 88 62
Polygon -7500403 true true 195 150 285 150 270 225 210 225
Polygon -7500403 true true 210 225 195 300 225 300 240 255 255 300 285 300 270 225
Polygon -7500403 true true 195 150 165 225 180 225 210 180
Polygon -7500403 true true 285 150 300 180 300 225 255 165 285 150

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

none
false
0

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

sheep-gone
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
Polygon -2674135 true false 0 15 15 0 300 285 285 300 0 15
Polygon -2674135 true false 0 285 15 300 300 15 285 0

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

triangle-down
false
0
Polygon -7500403 true true 150 270 15 45 285 45

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
  <experiment name="full_sensit_analysis_TRAD" repetitions="100" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>surviving-households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean-livestock</metric>
    <metric>sum-livestock-end</metric>
    <metric>gini-coef-surviving</metric>
    <metric>descriptive-norm</metric>
    <steppedValueSet variable="number-households" first="50" step="10" last="100"/>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-TRAD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-SAT">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-MAX">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-random">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;TRAD&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="w">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr1">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr2">
      <value value="0.1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0.8"/>
    </enumeratedValueSet>
    <steppedValueSet variable="intrinsic-preference" first="0" step="0.05" last="1"/>
    <steppedValueSet variable="social-susceptibility" first="0" step="0.05" last="1"/>
    <enumeratedValueSet variable="satisficing-threshold">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-trials">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting-threshold">
      <value value="0.2"/>
      <value value="0.4"/>
      <value value="0.6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="start-on-homepatch?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-mean">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-std">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="global-rain?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="full_sensit_analysis_SAT" repetitions="100" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>surviving-households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean-livestock</metric>
    <metric>sum-livestock-end</metric>
    <metric>gini-coef-surviving</metric>
    <metric>descriptive-norm</metric>
    <steppedValueSet variable="number-households" first="50" step="10" last="100"/>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-TRAD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-SAT">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-MAX">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-random">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;SAT&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="w">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr1">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr2">
      <value value="0.1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="intrinsic-preference">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-susceptibility">
      <value value="0"/>
    </enumeratedValueSet>
    <steppedValueSet variable="satisficing-threshold" first="10" step="5" last="150"/>
    <steppedValueSet variable="satisficing-trials" first="5" step="5" last="100"/>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting-threshold">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="start-on-homepatch?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-mean">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-std">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="global-rain?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="sensit_analysis_SAT_threshold" repetitions="100" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print ( word ( behaviorspace-run-number / 810000) " %" )</final>
    <metric>surviving-households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean-livestock</metric>
    <metric>sum-livestock-end</metric>
    <metric>gini-coef-surviving</metric>
    <metric>descriptive-norm</metric>
    <steppedValueSet variable="number-households" first="20" step="1" last="100"/>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-TRAD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-SAT">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-MAX">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-random">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;SAT&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="w">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr1">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr2">
      <value value="0.1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="intrinsic-preference">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-susceptibility">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-threshold">
      <value value="80"/>
    </enumeratedValueSet>
    <steppedValueSet variable="satisficing-trials" first="1" step="1" last="100"/>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting-threshold">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="start-on-homepatch?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-mean">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-std">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="global-rain?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="demogrChange_MAX" repetitions="100" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>surviving-households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean-livestock</metric>
    <metric>sum-livestock-end</metric>
    <metric>gini-coef-surviving</metric>
    <metric>descriptive-norm</metric>
    <steppedValueSet variable="number-households" first="20" step="1" last="100"/>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-TRAD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-SAT">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-MAX">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-random">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;MAX&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="w">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr1">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr2">
      <value value="0.1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="intrinsic-preference">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-susceptibility">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-threshold">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-trials">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting-threshold">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="start-on-homepatch?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-mean">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-std">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="global-rain?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="demogrChange_SAT" repetitions="100" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>surviving-households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean-livestock</metric>
    <metric>sum-livestock-end</metric>
    <metric>gini-coef-surviving</metric>
    <metric>descriptive-norm</metric>
    <steppedValueSet variable="number-households" first="20" step="1" last="100"/>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-TRAD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-SAT">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-MAX">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-random">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;SAT&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="w">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr1">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr2">
      <value value="0.1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="intrinsic-preference">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-susceptibility">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-threshold">
      <value value="50"/>
      <value value="80"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-trials">
      <value value="10"/>
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting-threshold">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="start-on-homepatch?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-mean">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-std">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="global-rain?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="demogrChange_TRAD_02" repetitions="100" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>surviving-households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean-livestock</metric>
    <metric>sum-livestock-end</metric>
    <metric>gini-coef-surviving</metric>
    <metric>descriptive-norm</metric>
    <steppedValueSet variable="number-households" first="20" step="1" last="100"/>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-TRAD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-SAT">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-MAX">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-random">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;TRAD&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="w">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr1">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr2">
      <value value="0.1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="intrinsic-preference">
      <value value="0.95"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-susceptibility">
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-threshold">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-trials">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting-threshold">
      <value value="0.2"/>
      <value value="0.4"/>
      <value value="0.6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="start-on-homepatch?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-mean">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-std">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="global-rain?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="demogrChange_TRAD_08" repetitions="100" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>surviving-households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean-livestock</metric>
    <metric>sum-livestock-end</metric>
    <metric>gini-coef-surviving</metric>
    <metric>descriptive-norm</metric>
    <steppedValueSet variable="number-households" first="20" step="1" last="100"/>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-TRAD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-SAT">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-MAX">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-random">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;TRAD&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="w">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr1">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr2">
      <value value="0.1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="intrinsic-preference">
      <value value="0.95"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-susceptibility">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-threshold">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-trials">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting-threshold">
      <value value="0.2"/>
      <value value="0.4"/>
      <value value="0.6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="start-on-homepatch?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-mean">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-std">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="global-rain?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="multiple-agents-homog-variance-analysis-lowf" repetitions="5000" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean [livestock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>gini-coef</metric>
    <metric>count households with [household-SL? = true]</metric>
    <metric>mean [livestock-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;E-RO&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-households">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="multiple-agents-homog-variance-analysis-highf" repetitions="1000" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean [livestock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>gini-coef</metric>
    <metric>count households with [household-SL? = true]</metric>
    <metric>mean [livestock-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;E-LBD&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-households">
      <value value="95"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-3types-homog-lowf-1000" repetitions="1000" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean [livestock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>gini-coef</metric>
    <metric>count households with [household-SL? = true]</metric>
    <metric>mean [livestock-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>gini-coef-surviving</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;E-RO&quot;"/>
      <value value="&quot;E-LBD&quot;"/>
      <value value="&quot;E-RO-SL1&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-households">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-one-agent-sensitivity" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att] of households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;E-RO&quot;"/>
      <value value="&quot;E-LBD&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-households">
      <value value="1"/>
    </enumeratedValueSet>
    <steppedValueSet variable="livestock-init" first="10" step="5" last="1500"/>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <steppedValueSet variable="risk-att-init" first="-0.9" step="0.1" last="0.9"/>
  </experiment>
  <experiment name="New-multiple-agents-sens-r-hh-init" repetitions="1000" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;E-LBD&quot;"/>
      <value value="&quot;E-RO&quot;"/>
      <value value="&quot;E-RO-SL1&quot;"/>
    </enumeratedValueSet>
    <steppedValueSet variable="number-households" first="5" step="10" last="95"/>
    <enumeratedValueSet variable="timesteps">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-multiple-agents-homog-sensitivity" repetitions="50" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att-init] of households</metric>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>gini-coef</metric>
    <metric>count households with [household-SL? = true]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [livestock + destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [livestock + destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type = "E-RO"]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [destock-total] of households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>mean [destock-total] of households with [household-behavioral-type = "E-RO"]</metric>
    <metric>mean [destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>gini-coef-surviving</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;E-RO&quot;"/>
      <value value="&quot;E-LBD&quot;"/>
      <value value="&quot;E-RO-SL1&quot;"/>
    </enumeratedValueSet>
    <steppedValueSet variable="number-households" first="5" step="10" last="95"/>
    <enumeratedValueSet variable="livestock-init">
      <value value="80"/>
      <value value="125"/>
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-multiple-agents-homog-FULLm-sens" repetitions="1000" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att-init] of households</metric>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att-init] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>gini-coef</metric>
    <metric>count households with [household-SL? = true]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [livestock + destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [livestock + destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type = "E-RO"]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [destock-total] of households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>mean [destock-total] of households with [household-behavioral-type = "E-RO"]</metric>
    <metric>mean [destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>gini-coef-surviving</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;E-LBD&quot;"/>
      <value value="&quot;E-RO&quot;"/>
      <value value="&quot;E-RO-SL1&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-households">
      <value value="5"/>
      <value value="30"/>
      <value value="50"/>
      <value value="70"/>
      <value value="95"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="70"/>
      <value value="75"/>
      <value value="90"/>
      <value value="110"/>
      <value value="160"/>
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-multiple-agents-SL2-switch-LBD" repetitions="1000" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>count households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO-SL1"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO"]</metric>
    <metric>count households with [household-SL? = TRUE]</metric>
    <metric>count households with [household-SL2? = TRUE]</metric>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [livestock-healthy-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [any? households-here]</metric>
    <metric>mean [green-biomass] of patches with [any? households-here]</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <enumeratedValueSet variable="number-households">
      <value value="15"/>
      <value value="45"/>
      <value value="85"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-LBD">
      <value value="1"/>
      <value value="3"/>
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO-SL1">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-multiple-agents-SL2-switch-SL1" repetitions="1000" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>count households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO-SL1"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO"]</metric>
    <metric>count households with [household-SL? = TRUE]</metric>
    <metric>count households with [household-SL2? = TRUE]</metric>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [livestock-healthy-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [any? households-here]</metric>
    <metric>mean [green-biomass] of patches with [any? households-here]</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <enumeratedValueSet variable="number-households">
      <value value="15"/>
      <value value="45"/>
      <value value="85"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO-SL1">
      <value value="1"/>
      <value value="3"/>
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-LBD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-multiple-agents-SL2-switch-eq" repetitions="1000" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>count households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO-SL1"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO"]</metric>
    <metric>count households with [household-SL? = TRUE]</metric>
    <metric>count households with [household-SL2? = TRUE]</metric>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [livestock-healthy-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [any? households-here]</metric>
    <metric>mean [green-biomass] of patches with [any? households-here]</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <enumeratedValueSet variable="number-households">
      <value value="3"/>
      <value value="9"/>
      <value value="15"/>
      <value value="30"/>
      <value value="60"/>
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO-SL1">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-LBD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-multiple-agents-homog-partm-knRAD" repetitions="1000" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>gini-coef</metric>
    <metric>count households with [household-SL? = true]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-LBD"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [livestock + destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [livestock + destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [destock] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type = "E-RO"]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [destock-total] of households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>mean [destock-total] of households with [household-behavioral-type = "E-RO"]</metric>
    <metric>mean [destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [destock-total] of households with [(household-behavioral-type = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>gini-coef-surviving</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;E-RO-SL1&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-households">
      <value value="5"/>
      <value value="50"/>
      <value value="95"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="2"/>
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-hetSL2on-hetSL2off-hh60-li90-v2" repetitions="15" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att-init] of households</metric>
    <metric>mean [household-risk-att] of households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <metric>count households with [household-SL? = true]</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>count households with [household-behavioral-type = "E-RO"]</metric>
    <enumeratedValueSet variable="number-households">
      <value value="60"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-LBD">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO-SL1">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="true"/>
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="New-hetSL2on-hetSL2off-hh60-li90-v2-INIT" repetitions="1000" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <final>print behaviorspace-run-number</final>
    <metric>mean [household-risk-att-init] of households</metric>
    <metric>mean [household-risk-att] of households</metric>
    <metric>[household-risk-att-init] of households</metric>
    <metric>[household-risk-att] of households</metric>
    <metric>mean-reserve-biomass</metric>
    <metric>mean-green-biomass</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>gini-coef</metric>
    <metric>count households with [household-SL? = true]</metric>
    <metric>count households with [household-SL2? = true]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>mean [reserve-biomass] of patches with [any? households-here]</metric>
    <metric>mean [green-biomass] of patches with [any? households-here]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type-init = "E-LBD"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type-init = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type-init = "E-LBD"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type-init = "E-RO"] ]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-behavioral-type-init = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-behavioral-type-init = "E-RO-SL1"] ]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-SL2? = TRUE]]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [household-SL2? = FALSE]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-SL2? = TRUE]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [household-SL2? = FALSE]]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = TRUE)]]</metric>
    <metric>mean [reserve-biomass] of patches with [ any? households-here with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [green-biomass] of patches with [ any? households-here with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = FALSE)]]</metric>
    <metric>mean [livestock + destock] of households with [household-behavioral-type-init = "E-LBD"]</metric>
    <metric>mean [livestock + destock] of households with [household-behavioral-type-init = "E-RO"]</metric>
    <metric>mean [livestock + destock] of households with [household-behavioral-type-init = "E-RO-SL1"]</metric>
    <metric>mean [livestock + destock] of households with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [livestock + destock] of households with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [livestock + destock] of households with [household-SL2? = TRUE]</metric>
    <metric>mean [livestock + destock] of households with [household-SL2? = FALSE]</metric>
    <metric>mean [destock] of households with [household-behavioral-type-init = "E-LBD"]</metric>
    <metric>mean [destock] of households with [household-behavioral-type-init = "E-RO"]</metric>
    <metric>mean [destock] of households with [household-behavioral-type-init = "E-RO-SL1"]</metric>
    <metric>mean [destock] of households with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [destock] of households with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [destock] of households with [household-SL2? = TRUE]</metric>
    <metric>mean [destock] of households with [household-SL2? = FALSE]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type-init = "E-LBD"]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type-init = "E-RO"]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-behavioral-type-init = "E-RO-SL1"]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-SL2? = TRUE]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households with [household-SL2? = FALSE]</metric>
    <metric>mean [destock-total] of households with [household-behavioral-type-init = "E-LBD"]</metric>
    <metric>mean [destock-total] of households with [household-behavioral-type-init = "E-RO"]</metric>
    <metric>mean [destock-total] of households with [household-behavioral-type-init = "E-RO-SL1"]</metric>
    <metric>mean [destock-total] of households with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = TRUE)]</metric>
    <metric>mean [destock-total] of households with [(household-behavioral-type-init = "E-RO-SL1") AND (household-SL? = FALSE)]</metric>
    <metric>mean [destock-total] of households with [household-SL2? = TRUE]</metric>
    <metric>mean [destock-total] of households with [household-SL2? = FALSE]</metric>
    <metric>gini-coef-surviving</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <metric>count households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO-SL1"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO"]</metric>
    <enumeratedValueSet variable="number-households">
      <value value="60"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-LBD">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO-SL1">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="true"/>
      <value value="false"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="experiment" repetitions="50" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <exitCondition>ticks = 1000</exitCondition>
    <metric>mean [household-intrinsic-preference] of households</metric>
    <metric>mean [household-social-susceptibility] of households</metric>
    <metric>mean [reserve-biomass] of patches</metric>
    <metric>mean [green-biomass] of patches</metric>
    <metric>mean [livestock + destock] of households</metric>
    <metric>mean [destock] of households</metric>
    <metric>count households with [household-SL? = true]</metric>
    <metric>count households with [household-SL2? = true]</metric>
    <metric>mean [livestock-healthy-total + destock-total] of households</metric>
    <metric>mean [destock-total] of households</metric>
    <metric>gini-coef-total-livestock-healthy</metric>
    <metric>count households with [household-behavioral-type = "E-LBD"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO-SL1"]</metric>
    <metric>count households with [household-behavioral-type = "E-RO"]</metric>
    <enumeratedValueSet variable="behavioral-type">
      <value value="&quot;E-RO&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-SAT">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-TRAD">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="b">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-MAX">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-mean">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="homog-behav-types?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-att-init">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="resting-threshold">
      <value value="0.1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr1">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-LBD">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-threshold">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="gr2">
      <value value="0.1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-E-RO-SL1">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="timesteps">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="w">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="start-on-homepatch?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="intrinsic-preference">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="SL2-strategy-switch?">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-random">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="number-households">
      <value value="60"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="extension-model?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="livestock-init">
      <value value="90"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="risk-mode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="satisficing-trials">
      <value value="21"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="rain-std">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="global-rain?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="social-susceptibility">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="knowledge-radius">
      <value value="1"/>
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
1
@#$#@#$#@
