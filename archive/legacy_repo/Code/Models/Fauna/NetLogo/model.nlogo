extensions [csv]
; modified from the patch choice model from optimal foraging theory
; (Michael Barton, Arizona State University)
; by Isaac Ullah and Miriam Coleman-Kopels, San Diego State University

breed [foragers forager] ;;foragers are bands of human hunter-gatherers who search for, kill, and consume animals
breed [animals animal] ;;animals are prey animals that consume grass and are prey for foragers
breed [kills kill] ;;kills are animals that have been killed by hunters. Needed only for visualiztion of kills locations

foragers-own [energy diet-breadth]
;;forager energy is the budget for movement costs, and replenshed by succesfull kills
;;forager diet-breadth is a continually updated account of the different types of prey animals recently consumed
animals-own [energy age species payoff rank time-since-repro]
;;animal energy is the budget for movement costs, and replenshed by grazing grass
;;animal age is the number of ticks an animal has been alive
;;animal species is one of male or female
;;animal payoff is the energy gain to foragers if animal is killed and consumed
;;animal rank is the rank of that species in a basic optimal foraging balancing equation (1 or 2)
;;animal time-since-repro is a count of ticks since last reproduction. For female animals only.
patches-own [stimer gtimer desert]
;;patches stimer is a count of ticks since the previous change from wet to dry season
;;patches stimer is a count of ticks since last grazing event
;;patches desert is a boolean value that indicates if a patch will stay barren (1) or will allow vegetation growth (0)
kills-own [ktimer]
;;kills ktime is tick count for lenght of display of the location of a kill
globals [prey-list long-prey-list diversity _recording-save-file-name birth-count death-count num-foragers num-animals males-killed females-killed ppfem-killed seasonal wet-season dry-season]
;;prey-list is a list of prey animals taken within a moving window of ticks
;;long-prey-list is a complete list of all prey animals taken during the simulation run
;;diversity is a continually updated measure of the variety of prey species taken (number of different species) within a rolling window of ticks. Relates to prey-list.
;;_recording-save-file-name is the file name for output csv stats
;;birth-count is a continually updated count of animal births
;;death-count is a continually updated count of animal deaths
;;num-foragers is a continually updated count of forager bands
;;num-animals is a continually updated count of animals
;;males-killed is a continually updated count of male animal kills
;;females-killed is a continually updated count of female animal kills
;;ppfem-killed is a continually updated count of female animal kills when the animal was pregnant or with calf
;;seasonal is a boolean indicating if we are modeling seasonality or continual regrowth
;;wet-season is a boolean indicating that the current season is the growing season
;;dry-season is a boolean indicating that the current season is the non-growing season



to Setup
  ;;initate a new world and instantiate patch charateristics and animal and forager turtles
  clear-all
  Setup_Animals
  Setup_Foragers
  Setup_Patches
  set males-killed 0
  set females-killed 0
  set ppfem-killed 0
  set num-foragers count foragers
  set num-animals count animals
  if not empty? Output-csv-file [
    file-open Output-csv-file
    file-print csv:to-row [ "Males Taken" "Females Taken" "Post Partum Femals Taken" "Forager Energy" "Number Males" "Number Females" "Buffalo Energy" "Buffalo Births" "Buffalo Deaths" "Proportion Ungrazed" ]
  ]
  reset-ticks
end

to Go
  ;;advance one tick and initiate actions for current tick
  if init-foragers != 0 [
    if not any? foragers [stop]
  ]
  if not any? animals [stop]
  if ( stop-at-max-ticks = true ) and ticks >= max-ticks [stop]

  ask foragers [
    set energy energy - foragers-movement-cost ; apply movement costs
    ifelse food-storage = true
      [ if energy < 100 [ Move Forage ] ] ; foragers don't move if stored food is available
      [ Move Forage]
    Calculate-Diversity
    if imortal-foragers = false [
      Check-Death
      ]
    ]

  ask animals [
    set age age + 1 ; animals get older
    Move-animals
    set energy energy - animals-movement-cost ; apply movement costs
    eat-grass
    if [species] of self = 2 [ ; am I female?
      set time-since-repro time-since-repro + 1 ; advance the reproduction clock
    ]
    ifelse stop-at-max-herd-size = false [ ; check to see if mating is allowed currently, mate if so
      mate
      ]
      [
      if count animals < max-herd-size [
        mate ]
    ]
    if energy >= 100 [ set energy 100 ] ;; make sure that energy does not go above maximum
    Check-Animal-Death
    if [species] of self = 2 and [time-since-repro] of self = birth-spacing [
      wean
      set size 4
      set color pink
    ]
    ]

  ask kills [ ; these are just icons of where kills occured
    set ktimer ktimer - 1
    if ktimer <= 0 [die] ; icon disappears after a few ticks
  ]
  if grass-growth = true [ ask patches [ grow-grass ] ]

  Do_Plots
  if not empty? Output-csv-file [
    output-data
  ]
  tick
end

to Setup_Patches
  ;;initiate patch charateristics
  ifelse seasonality = 1.0 [set seasonal false] [set seasonal true] ;check if we are doing seasonal or continual regrowth
  ask patches [ set pcolor green ]
  if grass-growth = true [ ask up-to-n-of ( ( count patches ) * (1 - grass-proportion ) ) patches [ set pcolor brown ] ] ; if variable grass, set patches up
  ifelse seasonal = true [
    set wet-season round(ticks-per-year * seasonality) ; set up seasons according to sliders
    set dry-season ticks-per-year - wet-season
    ask patches [
      set stimer ticks-per-year ; this is the seasonality timer
      ifelse pcolor = green
        [ set gtimer 0]
        [ set gtimer random regrowth-time ] ; initialize grass regrowth clocks randomly for brown patches
    ]
  ]
  [
    ask patches [
    set gtimer random regrowth-time ; set the intital regrowth timer for all patches
    ]
  ]
  if keep-initial-grass-pattern = true [
    ask patches [ if pcolor = brown [ set desert true ] ] ; if keep-initial is on, inital brown patches will stay brown
  ]
end

to Setup_Foragers
  ;;initiate forager characteristics
  create-foragers init-foragers
    [
    set shape "hunter2"
    set size 5
    set color 38
    set energy 100 ;; FUTURE ENHANCEMENT: make this a slider
    set prey-list [] ; rolling list of prey species taken
    set long-prey-list [] ; a longer list for all kills
    ]
  ask foragers [setxy random-xcor random-ycor] ; place the foragers randomly in the world
end

to Setup_Animals
  ; Create males and females at equal initial density
  let number-of-males round (init-prey * 0.5)
  let number-of-females round (init-prey * 0.5)

  set birth-count 0
  set death-count 0

  create-animals number-of-males [ ; These are Males
    setxy 0.5 * max-pxcor 0.5 * max-pycor
    set species 1
    set energy 100
    set age random age-of-senescence ; set up an initial age profile
    set shape "cow"
    set size 6
    set color cyan
    set payoff ( food-value-males - processing-cost-males ) ; find payoff of food value minus processing costs
    ifelse payoff >= ( food-value-females - processing-cost-females ) ; set rank of prey
      [ set rank 1 ]
      [ set rank 2 ]
    ]

  create-animals number-of-females [ ; These are Females
    setxy 0.5 * max-pxcor 0.5 * max-pycor
    set species 2
    set energy 100
    set age random age-of-senescence ; set up an initial age profile
    set time-since-repro 0 ; only females get this
    set shape "cow"
    set size 4
    set color pink
    set payoff ( food-value-females - processing-cost-females ) ; find payoff of food value minus processing costs
    ifelse payoff >= ( food-value-males - processing-cost-males ) ; set rank of prey
      [ set rank 1 ]
      [ set rank 2 ]
    ]

   ask animals [ ; initial setup to minimize "burn in" time of model runs
    fd (random (0.25 * max-pxcor)) ; move the animals into a loose cluster
    set heading mean-heading [ heading ] of animals in-radius (0.25 * max-pxcor) ; get them facing the same way as neighbors
  ]
end

to mate
  ;;allow female animal a chance to get pregnant and give birth
  let mates one-of animals-on neighbors ; pick a random nearby animal
  if mates != nobody  [    ; did we get someone?
    if ([species] of mates = 1) and ([species] of self = 2) and ([time-since-repro] of self >= birth-spacing) [ ; am I female? Did I meet a male? and has it been long enough?
        if random 100 <= energy [  ; reproduction is probabalistic and linked to energy status
          set birth-count birth-count + 1
          set time-since-repro 0
          set size 6
        ]
      ]
  ]
end

to wean
  ;;separate offspring turtle from mother. Offspring is created as an independent animal turtle and given characteristics.
  hatch 1 [
    ifelse random 100 <= 50 [
      set species 1 ; spawn a male and give it some male attributes
      set shape "cow"
      set size 6
      set color cyan
      set payoff ( food-value-males - processing-cost-males ) ; find payoff of food value minus processing costs
      ifelse payoff >= ( food-value-females - processing-cost-females ) ; set rank of prey
        [ set rank 1 ]
        [ set rank 2 ]
      set energy 100
      set age 0
    ]
    [ set species 2 ; spawn a female and give it some female attributes
      set shape "cow"
      set size 4
      set color pink
      set payoff ( food-value-females - processing-cost-females ) ; find payoff of food value minus processing costs
      ifelse payoff >= ( food-value-males - processing-cost-males ) ; set rank of prey
        [ set rank 1 ]
        [ set rank 2 ]
      set energy 100
      set age 0
      set time-since-repro 0 ; only females get this
    ]
  ]
end

to Move
  ;;forager agent movement logic
  let target min-one-of (animals in-radius foraging-radius) with [rank = 1] [distance myself] ; closest high-rank prey in radius
  let alt-target min-one-of (animals in-radius foraging-radius) with [species = 2 and time-since-repro <= birth-spacing] [distance myself] ; closest post partum female
  ifelse target != nobody and alt-target != nobody [ ;are there both kinds of prey?
    ifelse [payoff] of alt-target + food-value-calf > [payoff] of target [ ;pp female value is higher
      face alt-target
      fd 1 ] [
      face target
      fd 1
    ]
  ] [
  ifelse target != nobody  [
    face target ; face the hunting target and walk that direction
    fd 1
  ] [
    rt random 45
    lt random 45 ; otherwise do a random walk
    fd 1
  ] ]
end

to Move-animals
  ;;animal agent movement logic
  let target-patch min-one-of (patches in-radius 3 with [pcolor = green]) [distance myself] ; First, look for nearby grass...
  ifelse target-patch != nobody  [
    face target-patch ; if so, then move towards it...
    fd 1
  ] [
    let target-anims (animals in-radius 5)
    ; ...if not, then look for any nearby buffalo...
    ifelse target-anims != nobody  [
      set heading mean-heading [ heading ] of target-anims ; ... if so, then face the average way your herd neighbors are facing...
      fd 1
    ] [ set heading random 360 fd 1 ] ; ...if not, then do a random walk.
  ]
end

to Forage
  ;;forager will attempt to pursue, kill, and consume a highly-ranked nearby prey animal. Updates forager energy if successful, and updates animal population.
  let last-count count animals
  let prey one-of animals-here  ;; encounter a random nearby animal
  let other-prey animals with [rank = 1] in-radius foraging-radius ;; give forager knowledge ofhigh-ranked animals within set distance
  let alternative-payoff 0
  if count other-prey != 0 [
    set alternative-payoff (mean [payoff] of other-prey) - ( (count patches in-radius foraging-radius / count other-prey ) * foragers-movement-cost)
  ]
  if prey != nobody  [ ;; did we get one?  If so,
    let current-payoff 0  ;; current payoff of encountered animal
    ifelse [species] of prey = 2 and [time-since-repro] of prey <= birth-spacing
      [ set current-payoff [payoff] of prey + food-value-calf ]  ;; if prey is female and recently calved, add food value for calf with mother
      [ set current-payoff [payoff] of prey ];; males and other femals get assigned payoff
    if (current-payoff >= alternative-payoff) or  ;; only pursue prey with payoff greater than continued search,
       (energy <= alternative-payoff)  ; forager is starving and pursues whatever prey is encountered
        [ set energy energy + current-payoff  ;; get energy from eating animal
          set prey-list fput ([species] of prey) prey-list ; add prey-species to running list of prey taken
          set long-prey-list fput ([species] of prey) long-prey-list ; add prey-species to running list of prey taken
          if [species] of prey = 1 [set males-killed males-killed + 1]
          if [species] of prey = 2 [set females-killed females-killed + 1]
          if [species] of prey = 2 and [time-since-repro] of prey <= birth-spacing [
            set ppfem-killed ppfem-killed + 1 ; if a post partum female was killed, update that count.
          ]
        ]
        ask prey [ die ]                            ; kill it, and...
        hatch-kills 1 [
          set shape "cow"
          set size 7
          set color black
          set ktimer round(ticks-per-year / 12) ]
        hatch-kills 1 [
          set shape "x"
          set size 5
          set color red
          set ktimer round(ticks-per-year / 12) ]
      ]
  while [length prey-list > ticks-per-year] [set prey-list remove-item ticks-per-year prey-list] ; manage running list of prey taken
  if food-storage = false [
     if energy > 100 [ set energy 100 ] ;; make sure that energy does not go above maximum
  ]
  let this-count count animals
  set death-count (death-count + (last-count - this-count)) ; update running counts
end

to eat-grass
  ;;animal agent will attempt to consume a vegetated patch. Updates patch characteristics if grazed and animal energy.
  ifelse grass-growth = true [
  ; buffalo eat grass and turn the patch brown
  if pcolor = green [
    set pcolor brown
    set gtimer regrowth-time
    set energy energy + animals-gain-from-food  ; buffalo maintain energy by eating
    ]
  ]
  [
  ; buffalo graze, but grass stays green
  if pcolor = green [
    set energy energy + animals-gain-from-food  ; buffalo maintain energy by eating
    ]
  ]
end

to grow-grass
  ;;regrow grazed grass according to seasonality logic
  ifelse seasonal = true [
  ; update seasonal and regrowth timers
  set stimer stimer - 1
  set gtimer gtimer - 1
   if stimer > dry-season [ ; if we are in the wet-season, regrow grass continually. When in dry season, no regrowth will happen
     if pcolor = brown and gtimer <= 0
       [ set pcolor green ]
     if desert = true
       [ set pcolor brown ]; if desertification is on, then the initial brown patches stay brown
   ]
   if stimer <= 0
    [set stimer ticks-per-year] ; start a new year
  ]
  [
   ; continual reset timer
   set gtimer gtimer - 1
   if pcolor = brown [
     ifelse gtimer <= 0
       [ set pcolor green
         set gtimer regrowth-time ]
       [ set gtimer gtimer - 1 ]
     if desert = true [
      set pcolor brown ; if desertification is on, then the initial brown patches stay brown
     ]
   ]
  ]

end

to Calculate-Diversity
  ;; calcualte the number of species that has recently been consumed
  set diversity 0
  if member? 1 prey-list [set diversity diversity + 1]
  if member? 2 prey-list [set diversity diversity + 1]
end

to-report grass
  ;;count patches that are still vegetated
  ifelse grass-growth = true [
    report patches with [pcolor = green]
  ]
  [ report 0 ]
end

to-report mean-heading [ headings ]
  ;;calculate average direction of turtle movement
  let mean-x mean map sin headings
  let mean-y mean map cos headings
  report atan mean-x mean-y
end

to Do_Plots
  ;;update plots on graphica interface
  set-current-plot "Prey Taken"
  set-current-plot-pen "Male"
  plot males-killed
  set-current-plot-pen "Female"
  plot females-killed
  set-current-plot-pen "PP-Female"
  plot ppfem-killed
  set-current-plot "Average Energy"
  set-current-plot-pen "Foragers"
  ifelse count foragers != 0
    [plot (mean [energy] of foragers)]
    [plot 0]
  set-current-plot-pen "Animals"
  ifelse count animals != 0
    [plot (mean [energy] of animals)]
    [plot 0]
  set-current-plot "Population"
  set-current-plot-pen "Male"
  plot count animals with [species = 1]
  set-current-plot-pen "Female"
  plot count animals with [species = 2]
  set-current-plot-pen "Foragers"
  plot num-foragers
  set-current-plot "Proportion of Grazed to Ungrazed Grass"
  set-current-plot-pen "pgraze"
  plot ( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100
end

to output-data
  ;;write data from current tick to csv file
  file-print csv:to-row ( list
    ( num-foragers )
    ( sum [energy] of foragers )
    ( males-killed )
    ( females-killed )
    ( ppfem-killed )
    ( birth-count )
    ( death-count )
    ( count animals with [species = 1] )
    ( count animals with [species = 2] )
    ( count animals )
    ( sum [energy] of animals with [species = 1] )
    ( sum [energy] of animals with [species = 2] )
    ( sum [energy] of animals )
    ( ( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100 )
      )
  file-flush
end

to Check-Death
  ;;mortality procedure for forager agents.
  ;if foragers can starve, then they can die if starving
  ask foragers [if energy <= 0 [die]]
  set num-foragers count foragers
end

to Check-Animal-Death
  ;;mortality proceedure for animal agents
  ; Animals in hunger stress or that have lived longer than senescence have an increased chance of dying.
  ; Totally spent animals, and animals past maximum lifespan will always die.
  ; apply a mortality curve for external predation and accidental death
  let last-count count animals
  ask animals with [energy <= 0] [die]
  ask up-to-n-of (last-count * (internal-mortality)) animals with [energy <= animals-starvation-threshold] [die]
  ask animals with [age > maximum-lifespan] [die]
  ask up-to-n-of (last-count * (internal-mortality)) animals with [age >= age-of-senescence] [die]
  ask up-to-n-of (last-count * (external-mortality)) animals [die]
  let this-count count animals
  set death-count (death-count + (last-count - this-count))
  set num-animals count animals
end
@#$#@#$#@
GRAPHICS-WINDOW
575
150
1099
675
-1
-1
2.45
1
10
1
1
1
0
1
1
1
0
210
0
210
1
1
1
ticks
40.0

BUTTON
234
148
300
181
setup
Setup
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
304
148
367
181
run
Go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
13
28
223
61
init-foragers
init-foragers
0
100
0.0
1
1
NIL
HORIZONTAL

SLIDER
192
196
371
229
processing-cost-males
processing-cost-males
0
20
3.0
1
1
NIL
HORIZONTAL

SLIDER
193
234
372
267
food-value-males
food-value-males
5
100
100.0
1
1
NIL
HORIZONTAL

BUTTON
374
148
437
181
step
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
10
345
222
378
maximum-lifespan
maximum-lifespan
0
12000
10220.0
1
1
NIL
HORIZONTAL

SLIDER
375
196
567
229
processing-cost-females
processing-cost-females
0
20
20.0
1
1
NIL
HORIZONTAL

SLIDER
375
233
568
266
food-value-females
food-value-females
5
100
100.0
1
1
NIL
HORIZONTAL

TEXTBOX
17
173
155
203
Animal Parameters
12
0.0
1

SLIDER
10
196
189
229
init-prey
init-prey
0
1000
250.0
1
1
NIL
HORIZONTAL

PLOT
10
433
561
558
Prey Taken
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
"Male" 1.0 0 -11221820 true "" ""
"Female" 1.0 0 -3508570 true "" ""
"PP-Female" 1.0 0 -8630108 true "" ""

MONITOR
474
367
543
412
Males
length (filter [ ?1 -> ?1 = 1 ] prey-list)
0
1
11

MONITOR
473
317
542
362
Females
length (filter [ ?1 -> ?1 = 2 ] prey-list)
0
1
11

TEXTBOX
475
282
549
309
# taken over last year
10
0.0
1

PLOT
10
685
560
805
Average Energy
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
"Foragers" 1.0 0 -16777216 true "" ""
"Animals" 1.0 0 -7500403 true "" ""

SLIDER
12
382
222
415
internal-mortality
internal-mortality
0
.2
0.2
0.0001
1
NIL
HORIZONTAL

SLIDER
10
308
222
341
age-of-senescence
age-of-senescence
0
10000
7300.0
1
1
NIL
HORIZONTAL

PLOT
574
685
1099
805
Population
NIL
Number
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"Male" 1.0 0 -11221820 true "" ""
"Female" 1.0 0 -3508570 true "" ""
"Foragers" 1.0 0 -16777216 true "" ""

SLIDER
740
35
905
68
ticks-per-year
ticks-per-year
0
720
365.0
1
1
NIL
HORIZONTAL

SLIDER
232
309
437
342
animals-gain-from-food
animals-gain-from-food
0
10
1.5
0.01
1
NIL
HORIZONTAL

SLIDER
233
347
436
380
animals-starvation-threshold
animals-starvation-threshold
0
100
25.0
1
1
NIL
HORIZONTAL

SLIDER
232
271
436
304
animals-movement-cost
animals-movement-cost
0
10
1.0
0.01
1
NIL
HORIZONTAL

SLIDER
14
65
224
98
foragers-movement-cost
foragers-movement-cost
0
10
1.0
.01
1
NIL
HORIZONTAL

PLOT
10
561
560
681
Proportion of Grazed to Ungrazed Grass
NIL
NIL
0.0
10.0
0.0
100.0
true
false
"" ""
PENS
"pgraze" 1.0 0 -14439633 true "" ""

INPUTBOX
420
20
540
80
Output-csv-file
NIL
1
0
String

INPUTBOX
1009
85
1098
145
max-herd-size
500.0
1
0
Number

INPUTBOX
913
85
1006
145
max-ticks
73000.0
1
0
Number

SWITCH
914
10
1097
43
stop-at-max-ticks
stop-at-max-ticks
0
1
-1000

SWITCH
914
48
1098
81
stop-at-max-herd-size
stop-at-max-herd-size
1
1
-1000

SWITCH
575
75
710
108
grass-growth
grass-growth
0
1
-1000

SLIDER
715
75
905
108
grass-proportion
grass-proportion
0
1
1.0
.01
1
NIL
HORIZONTAL

SWITCH
575
115
785
148
keep-initial-grass-pattern
keep-initial-grass-pattern
0
1
-1000

SLIDER
232
384
437
417
external-mortality
external-mortality
0
.2
0.2
0.0001
1
NIL
HORIZONTAL

MONITOR
460
99
552
144
Birth count
birth-count
0
1
11

SLIDER
11
272
222
305
birth-spacing
birth-spacing
0
1000
730.0
1
1
NIL
HORIZONTAL

SLIDER
14
101
224
134
foraging-radius
foraging-radius
1
100
6.0
1
1
NIL
HORIZONTAL

TEXTBOX
19
10
169
28
Forager Parameters
12
0.0
1

SLIDER
10
233
190
266
food-value-calf
food-value-calf
0
20
3.0
1
1
NIL
HORIZONTAL

SWITCH
234
67
409
100
imortal-foragers
imortal-foragers
0
1
-1000

MONITOR
460
147
552
192
Death count
death-count
0
1
11

MONITOR
235
15
315
60
# Foragers
num-foragers
17
1
11

TEXTBOX
650
8
849
38
Environmental Parameters
12
0.0
1

SWITCH
234
103
409
136
food-storage
food-storage
0
1
-1000

MONITOR
325
15
400
60
# Animals
num-animals
0
1
11

SLIDER
575
35
735
68
seasonality
seasonality
0
1
0.7
0.01
1
NIL
HORIZONTAL

SLIDER
790
115
905
148
regrowth-time
regrowth-time
0
365
3.0
1
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?
The Megafaunal Hunting Pressure Model (MHPM) is an interactive, agent-based model designed to conduct experiments to test megaherbivore extinction hypotheses. The model considers environmental change, human hunting behavior, herd demography, herd movement, animal sexual dimorphism, and animal life history.
## HOW IT WORKS
You first see a simplified grassland environment with randomly generated patches of grass, male and female ungulates, and human hunters. Depending on the hypotheses and/or data from a specific case study, you can manipulate the sliders in the NetLogo interface to establish the socio-natural dynamics of the experiment matrix. Broadly, the model allows you to change the following unitless categories that exist on a continuous scale: fodder availability, seasonality, relative animal food value, animal fecundity, lifespans of hunters and animals, relative hunting danger, as well as the energy required to move in the environment. Ideally, experimental dynamics will demonstrate how environmental conditions, human hunting, and animal life history interact to either accelerate or buffer extinction or create equilibrium conditions which do not result in an extinction event. The temporal units used in NetLogo are “ticks,” which can be scaled to any temporal granularity.
The landscape consists of an abstract square universe that wraps at the edges. It exists on a 211x211 patch matrix and a relative spatial scale. Although these patches are unitless, it may, be useful to image the landscape in km2 based on approximations of daily distance traveled by megaherbivores. For example, S. caffer walks approximately 3.4 km/day which scales a single patch to ~911 km2. These values will change for each case study.
## HOW TO USE IT
This is an interactive manual model. The user can fully manipulate the variables to test a wide range of hypotheses related to extinction in a socio-natural system. To begin first select the variables you would like to test in your extinction scenario (i.e., environmental patchiness or animal fecundity). We recommend keeping most variables constant, and only changing 1-3 variables per experiment. The user can also set the max ticks, which will stop the experiment after a designated number of ticks. Once the independent variables have been selected, use the interactive sliders to move them a few units at a time, avoiding large, drastic jumps in relative scale. When you are ready to run your experiment click the setup button. This will generate a landscape that matches the environment, herd demography, and hunting pressure reflected in the user’s parameterization. Then click the run button, and the experiment will begin.
Hunters are embedded with logic that approximates dynamic optimal foraging theory. In this model, hunter’s foraging decisions very in response to environmental feedback. The decision to pursue a prey item is balanced between the current energy-status of the hunter, the food value of the prey, the relative density of prey in the landscape, and the cost of pursuing, hunting, and processing the animal. Hunters do not reproduce and loose a user-set amount of energy as they move in their environment. Hunters can die if they run out of energy, they can also store energy from food and create an energy surplus, which is consumed before hunting is resumed. Users can turn human mortality on or off.
Ungulate herds are sexually dimorphic, with distinct user-determined food value and processing costs for males and females. Animals are also capable of reproduction, and sex ratios and probability of reproduction can be controlled by the user. Additionally, females can become pregnant and spawn offspring. Once spawned, offspring nurse for a set amount of time determined by the user. A nursing offspring adds to the food value of the female. This allows the user to approximate animal life history with novel detail. Animals are programmed to move as a herd with feedback from environmental conditions. Animals age, and the probability of random death increase once the animal reaches senescence. There is also external morality probability which can be adjusted by the user to simulate a percentage of animals that die from non-human predation at each time-step. Maximum animal lifespan is also a slider set by the user.
The environment can be controlled by the user by changing the density of fodder in the landscape (grass-proportion) and the time it takes for grass to regrow once it has been grazed (grass-regrowth). Grass regrowth can be turned off which will create a static environment. Additionally, users can choose to the keep-initial option, which limited grass regrowth only to the areas that were initially vegetated. Users can also utilize a seasonality slider, which limits grass regrowth to a user determined seasonal cycle.
The process of refining and experimenting with the relative value of independent and constant variables can be repeated until the experimental parameters appears to behave logically and work well to test the user’s hypotheses. When selecting values, it is useful to keep in mind that these values have relative, as opposed to absolute, value. For example, when picking a value for food-value-male and food-value-female (the energy hunter’s gain from eating male or female prey) do not think of the numbers in terms of kilogram or calories, but rather the relative size of males to females. In this case, a 2–5-unit difference may be sufficient to demonstrate how sexual dimorphism may impact human hunters operating under optimal foraging logic. The experiment will stop when: 1) human hunters die off, or 3) the experiment reaches the maximum number of ticks set by the user.
## SUMMARY STATISTICS
Summary statistics are shown on the bottom left side in labeled output boxes and are formatting as continuous line graphs that reflect new data from each tick. They include:
    • Number of prey-items taken per 10 ticks. Three lines are generated to show how many males, females, or post-partum females are hunted over the past 10-tick cycle.
    • Proportion of grazed to ungrazed grass on a scale of 0-100.
    • Forager energy, which changes as hunters exploit prey and move in their environment.
    • Number of animals in the environment. Three lines are generated to show how many males, females, or post-partum females are in the environment over time.
(More information about the NetLogo Behavior Space needed)
## THINGS TO TRY
To begin manual experiments prior to identifying final experimental parameters for the behavior space, establish a baseline understanding of extinction without the presence of hunters, when the environment remains static, and males and females have identical food-value. Once you have a baseline understanding of model dynamics, you can begin changing a few variables at a time to see if extinction dynamics change.
To examine how environment alone may impact extinction without human hunting, select one of the environmental variables – either grass-proportion or grass-regrowth – and move them a few units to represent degradation to baseline environmental conditions. Then click setup and run to observe how changes to environment might impact extinction dynamics. To continue this experiment, move the second environmental variable, to see how even greater environmental degradation impacts the extinction dynamics. Now re-run the same experiments with seasonality turned on, how does the presence of a ‘dry’ and ‘wet’ season impact extinction, herd movement, and herd demography? Do environmental conditions alone impact the likelihood of an extinction events? Identify a set of model parameters that test a range of environmental conditions (excellent – poor), with or without seasonality depending on your case study. Note that these differences between environmental conditions should represent only small differences between units – avoid drastic changes, which will make analysis very difficult.
To examine how hunting pressure and hunting behavior might impact extinction dynamics, begin by keeping the environment static. First decide on the population density of hunters. We recommend starting with a relatively low density of hunters, then increasing the number of hunters iteratively, adding a few hunters at a time, until the landscape best approximates your case study and/or hypotheses. Then, run your experiment without changing animal sexual dimorphism to see how hunting alone might impact extinction dynamics. Once you have a baseline understanding of how hunting might impact herd movement and demographics, you can begin creating a sexually dimorphic herd to observe how that might impact hunting behavior and the likelihood of extinction. Use data from your case study and/or hypotheses to select a sex that will be larger. Then, give that sex additional food-value (one or two additional units, which can be increased iteratively as needed). Now, click setup and run. How does sexual dimorphism impact hunting behavior? How does that behavior change the likelihood of extinction? What sex goes extinct first? This data will be available by observing the output in the lower left corner.
To examine how animal life history may impact extinct you can add to the hunting experiments above. Keep your herd sexually dimorphic and change the birth-spacing variable. Use data from your case study to inform the interbirth interval selected for the ungulate species. We recommend starting with a small number (for example 1 or 2) and increasing iteratively until the number accurately reflects your case study. Then use existing data to select a maximum lifespan for your animal. Once selected, we recommend keeping lifespan constant across all runs of the model. Then click setup and run. How does the interbirth interval (and the potential food value of the calf) impact hunting behavior? How does this impact extinction dynamics? Which sex is more likely to go extinct first?
Now you are ready to bring together environmental, hunting, and life history together into the experimental parameters. Begin by testing how hunting pressure and life history constraints play out in a ‘lush’ environment (with or without seasonality depending on your case study). Did the animals go extinct? What sex did hunters prefer? At what tick did extinction occur? Now, re-run that experiment with your ‘moderate’ environment, and then again with a poor environment. Compare the results between the three runs. When you are satisfied with the results you are ready to run your experiment in the behavior space to see how it behaves over hundreds of iterations.
## EXTENDING THE MODEL
As it exists, the landscape is a flat, rectangle with no topographical features. It would be useful to add GIS data so that specific case studies can be explored in greater detail. For example, how might environmental refugia impact – or buffer – the likelihood of extinction, like case studies seen in North America. To accurately test these hypotheses, the topographical features of the environmental must have more detail and plasticity.
## RELATED MODELS
The Diet Breadth Model is also a NetLogo model developed for testing how foraging decisions react dynamically to environmental feedback upon prey encounter. This model is simpler, and does not include environmental seasonality or animal life history.
## CREDITS AND REFERENCES
This model was created by Isaac I.T. Ullah, PhD, and Miriam C. Kopels, MA, at San Diego State University and The University of Nevada, Las Vegas. It is intended for educational and experimental purposes only and comes with no warrantee or guarantee of accuracy. The original idea for this model derived from Richard Klein’s case study of giant buffalo at the Klasies River Mouth, South Africa, during the Middle Stone Age. It is available under the GNU public license, and is published on the web at http://isaacullah.github.io/models/?????. Please cite the authors if you use this model in any publication.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

acorn
false
0
Polygon -7500403 true true 146 297 120 285 105 270 75 225 60 180 60 150 75 105 225 105 240 150 240 180 225 225 195 270 180 285 155 297
Polygon -6459832 true false 121 15 136 58 94 53 68 65 46 90 46 105 75 115 234 117 256 105 256 90 239 68 209 57 157 59 136 8
Circle -16777216 false false 223 95 18
Circle -16777216 false false 219 77 18
Circle -16777216 false false 205 88 18
Line -16777216 false 214 68 223 71
Line -16777216 false 223 72 225 78
Line -16777216 false 212 88 207 82
Line -16777216 false 206 82 195 82
Line -16777216 false 197 114 201 107
Line -16777216 false 201 106 193 97
Line -16777216 false 198 66 189 60
Line -16777216 false 176 87 180 80
Line -16777216 false 157 105 161 98
Line -16777216 false 158 65 150 56
Line -16777216 false 180 79 172 70
Line -16777216 false 193 73 197 66
Line -16777216 false 237 82 252 84
Line -16777216 false 249 86 253 97
Line -16777216 false 240 104 252 96

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

bird
false
0
Polygon -7500403 true true 135 165 90 270 120 300 180 300 210 270 165 165
Rectangle -7500403 true true 120 105 180 237
Polygon -7500403 true true 135 105 120 75 105 45 121 6 167 8 207 25 257 46 180 75 165 105
Circle -16777216 true false 128 21 42
Polygon -7500403 true true 163 116 194 92 212 86 230 86 250 90 265 98 279 111 290 126 296 143 298 158 298 166 296 183 286 204 272 219 259 227 235 240 241 223 250 207 251 192 245 180 232 168 216 162 200 162 186 166 175 173 171 180
Polygon -7500403 true true 137 116 106 92 88 86 70 86 50 90 35 98 21 111 10 126 4 143 2 158 2 166 4 183 14 204 28 219 41 227 65 240 59 223 50 207 49 192 55 180 68 168 84 162 100 162 114 166 125 173 129 180

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

caterpillar
true
0
Polygon -7500403 true true 165 210 165 225 135 255 105 270 90 270 75 255 75 240 90 210 120 195 135 165 165 135 165 105 150 75 150 60 135 60 120 45 120 30 135 15 150 15 180 30 180 45 195 45 210 60 225 105 225 135 210 150 210 165 195 195 180 210
Line -16777216 false 135 255 90 210
Line -16777216 false 165 225 120 195
Line -16777216 false 135 165 180 210
Line -16777216 false 150 150 201 186
Line -16777216 false 165 135 210 150
Line -16777216 false 165 120 225 120
Line -16777216 false 165 106 221 90
Line -16777216 false 157 91 210 60
Line -16777216 false 150 60 180 45
Line -16777216 false 120 30 96 26
Line -16777216 false 124 0 135 15

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

hunter
false
0
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 105 90 60 195 90 210 135 105
Polygon -7500403 true true 195 90 270 135 255 165 165 105
Circle -6459832 true false 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Polygon -6459832 true false 120 90 105 90 180 195 180 165
Line -6459832 false 109 105 139 105
Line -6459832 false 122 125 151 117
Line -6459832 false 137 143 159 134
Line -6459832 false 158 179 181 158
Line -6459832 false 146 160 169 146
Rectangle -6459832 true false 120 193 180 201
Rectangle -6459832 true false 114 187 128 208
Rectangle -6459832 true false 177 187 191 208
Polygon -16777216 true false 225 30 255 75 270 120 270 150 255 195 225 240 255 210 270 195 285 150 285 120 270 75 225 30

hunter2
false
0
Rectangle -7500403 true true 142 79 187 94
Polygon -7500403 true true 30 75 135 135 150 105 45 60
Polygon -7500403 true true 210 90 270 165 255 180 180 105
Circle -7500403 true true 125 5 80
Polygon -7500403 true true 120 90 135 195 105 300 150 300 135 285 165 225 180 300 225 300 210 285 195 195 210 90
Polygon -14835848 true false 135 90 120 90 195 195 195 165
Line -6459832 false 109 105 139 105
Line -6459832 false 122 125 151 117
Line -6459832 false 137 143 159 134
Line -6459832 false 158 179 181 158
Line -6459832 false 146 160 169 146
Polygon -14835848 true false 135 180 105 240 225 240 195 180
Rectangle -16777216 true false 132 178 199 188
Rectangle -6459832 true false 5 63 228 72
Polygon -11221820 true false 192 50 285 60 211 83

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

mouse side
false
0
Polygon -7500403 true true 38 162 24 165 19 174 22 192 47 213 90 225 135 230 161 240 178 262 150 246 117 238 73 232 36 220 11 196 7 171 15 153 37 146 46 145
Polygon -7500403 true true 289 142 271 165 237 164 217 185 235 192 254 192 259 199 245 200 248 203 226 199 200 194 155 195 122 185 84 187 91 195 82 192 83 201 72 190 67 199 62 185 46 183 36 165 40 134 57 115 74 106 60 109 90 97 112 94 92 93 130 86 154 88 134 81 183 90 197 94 183 86 212 95 211 88 224 83 235 88 248 97 246 90 257 107 255 97 270 120
Polygon -16777216 true false 234 100 220 96 210 100 214 111 228 116 239 115
Circle -16777216 true false 246 117 20
Line -7500403 true 270 153 282 174
Line -7500403 true 272 153 255 173
Line -7500403 true 269 156 268 177

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

rabbit
false
0
Polygon -7500403 true true 61 150 76 180 91 195 103 214 91 240 76 255 61 270 76 270 106 255 132 209 151 210 181 210 211 240 196 255 181 255 166 247 151 255 166 270 211 270 241 255 240 210 270 225 285 165 256 135 226 105 166 90 91 105
Polygon -7500403 true true 75 164 94 104 70 82 45 89 19 104 4 149 19 164 37 162 59 153
Polygon -7500403 true true 64 98 96 87 138 26 130 15 97 36 54 86
Polygon -7500403 true true 49 89 57 47 78 4 89 20 70 88
Circle -16777216 true false 37 103 16
Line -16777216 false 44 150 104 150
Line -16777216 false 39 158 84 175
Line -16777216 false 29 159 57 195
Polygon -5825686 true false 0 150 15 165 15 150
Polygon -5825686 true false 76 90 97 47 130 32
Line -16777216 false 180 210 165 180
Line -16777216 false 165 180 180 165
Line -16777216 false 180 165 225 165
Line -16777216 false 180 210 210 240

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

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.0
@#$#@#$#@
print random-float 1
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="2022-JQS-runs" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="12800"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="0"/>
      <value value="15"/>
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;output.csv&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-0-control" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;output.csv&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-10-0.66-0.8" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-10-0.66-0.4" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-10-0.33-0.4" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-10-0.33-0.8" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-20-3-0.66-0.8" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-20-3-0.66-0.4" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-20-3-0.33-0.4" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-20-3-0.33-0.8" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-environmental-sensitivity-sweep" repetitions="10" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0"/>
      <value value="0.1"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.6"/>
      <value value="0.7"/>
      <value value="0.8"/>
      <value value="0.9"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0"/>
      <value value="0.1"/>
      <value value="0.2"/>
      <value value="0.3"/>
      <value value="0.4"/>
      <value value="0.5"/>
      <value value="0.6"/>
      <value value="0.7"/>
      <value value="0.8"/>
      <value value="0.9"/>
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-15-6-0.66-0.8" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-15-6-0.66-0.4" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-15-6-0.33-0.4" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-15-6-0.33-0.8" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-20-6-0.66-0.8" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-20-6-0.66-0.4" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.66"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-20-6-0.33-0.4" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="2023-20-6-0.33-0.8" repetitions="40" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="73000"/>
    <metric>num-foragers</metric>
    <metric>sum [energy] of foragers</metric>
    <metric>males-killed</metric>
    <metric>females-killed</metric>
    <metric>ppfem-killed</metric>
    <metric>birth-count</metric>
    <metric>death-count</metric>
    <metric>count animals with [species = 1]</metric>
    <metric>count animals with [species = 2]</metric>
    <metric>count animals</metric>
    <metric>sum [energy] of animals with [species = 1]</metric>
    <metric>sum [energy] of animals with [species = 2]</metric>
    <metric>sum [energy] of animals</metric>
    <metric>( ( count patches with [ pcolor = green ] ) / ( count patches ) ) * 100</metric>
    <enumeratedValueSet variable="init-foragers">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-females">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="keep-initial-grass-pattern">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="maximum-lifespan">
      <value value="10220"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-males">
      <value value="30"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="internal-mortality">
      <value value="0.025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-ticks">
      <value value="73000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="external-mortality">
      <value value="0.0025"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-herd-size">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="init-prey">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foraging-radius">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="stop-at-max-ticks">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Output-csv-file">
      <value value="&quot;&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-calf">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-growth">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="imortal-foragers">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="grass-proportion">
      <value value="0.33"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="ticks-per-year">
      <value value="365"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="foragers-movement-cost">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-starvation-threshold">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="animals-gain-from-food">
      <value value="1.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="seasonality">
      <value value="0.8"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="processing-cost-males">
      <value value="6"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="age-of-senescence">
      <value value="7300"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="birth-spacing">
      <value value="730"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="max-herd-size">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-value-females">
      <value value="27"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="regrowth-time">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="food-storage">
      <value value="true"/>
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
