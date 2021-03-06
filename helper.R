library(foreign)
library(randomForest)

write_to_dta_file <- function(df, fname){
	write.dta(df, fname)
}

get_rf_feature_importance <- function(rf){
	feature_map <- c('s1', '1. HOW OLD ARE YOU?', 's10', '10. SCHOOL GIVES NO LETTER GRADES', 's10a',
		'10.1 GRADE IN ENGLISH', 's10b', '10.2 GRADE IN MATHEMATICS', 's10c', '10.3 GRADE IN HISTORY/SOCIAL STUDIES',
		's10d', '10.4 GRADE IN SCIENCE', 's11', '11. LIVES WITH MOTHER', 's12', '12. HOW FAR DID MOTHER GO IN SCHOOL?',
		's13', '13. MOTHER BORN IN U.S.?', 's16', '16. DOES MOTHER CARE ABOUT YOU?', 's17', '17. LIVES WITH FATHER',
		's18', '18. HOW FAR DID FATHER GO IN SCHOOL?', 's2', '2. WHAT SEX ARE YOU?', 's22',
		'22. DOES FATHER CARE ABOUT YOU?', 's26', '26. LIVE WITH BIOLOGICAL PARENTS?', 's3',
		'3. WHAT GRADE ARE YOU IN?', 's4', '4. ARE YOU OF HISPANIC/SPANISH ORIGIN?', 's44',
		'44. DOES NOT PART. ANY CLUBS,ORGS,TEAMS', 's45a', '45.a WILL LIVE TO AGE 35', 's45b',
		'45.b WILL MARRY BY AGE 25', 's45c', '45.c WILL BE KILLED BY AGE 21', 's45d',
		'45.d WILL GET HIV OR AIDS', 's45e', '45.e WILL GRADUATE FROM COLLEGE', 's45f',
		'45.f WILL HAVE MIDDLE CLASS INCOME', 's46a', '46.a TROUBLE GETTING ALONG WITH TEACHER',
		's46b', '46.b TROUBLE PAYING ATTENTION IN SCHOOL', 's46c', '46.c TROUBLE GETTING HOMEWORK DONE',
		's46d', '46.d TROUBLE WITH OTHER STUDENTS', 's47', '47. TIME SPENT WATCHING TV ON SCHOOL DAY',
		's48', '48. TRIES TO DO SCHOOL WORK WELL', 's49', '49. DRANK ALCOHOL MORE THAN 2/3 TIMES?', 's50',
		'50. HOW IS YOUR HEALTH?', 's59a', '59.a SMOKED CIGARETTES-LAST 12 MTHS', 's59b',
		'59.b DRANK BEER WINE LIQUOR-LAST 12 MTHS', 's59c', '59.c GOT DRUNK-LAST 12 MTHS', 's59d',
		'59.d RACED ON BIKE OR CAR-LAST 12 MTHS', 's59e', '59.e IN DANGER DUE TO DARE-LAST 12 MTHS',
		's59f', '59.f LIED TO PARENTS-LAST 12 MTHS', 's59g', '59.g SKIPPED SCHOOL-LAST 12 MTHS', 's60a',
		'60.a FELT SICK-LAST MONTH', 's60b', '60.b WOKE UP TIRED-LAST MONTH', 's60c',
		'60.c SKIN PROBLEMS-LAST MONTH', 's60d', '60.d DIZZY-LAST MONTH', 's60e', '60.e CHEST PAIN-LAST MONTH',
		's60f', '60.f HEADACHES-LAST MONTH', 's60g', '60.g SORE MUSCLES-LAST MONTH', 's60h',
		'60.h STOMACHACHE-LAST MONTH', 's60i', '60.i POOR APPETITE-LAST MONTH', 's60j',
		'60.j TROUBLE SLEEPING-LAST MONTH', 's60k', '60.k DEPRESSED-LAST MONTH', 's60l',
		'60.l TROUBLE RELAXING-LAST MONTH', 's60m', '60.m MOODY-LAST MONTH', 's60n',
		'60.n CRIED A LOT-LAST MONTH', 's60o', '60.o AFRAID OF THINGS-LAST MONTH',
		's61a', '61.a MISSED SCHOOL-HEALTH PROBLEM', 's61b', '61.b MISSED SOCIAL ACTIVITY-HEALTH PROB.',
		's61c', '61.c TROUBLE WALKING-HEALTH PROBLEM', 's61d', '61.d TROUBLE RUNNING-HEALTH PROBLEM',
		's61e', '61.e TROUBLE LIFTING-HEALTH PROBLEM', 's61f', '61.f TROUBLE WITH HANDS-HEALTH PROBLEM',
		's62a', '62.a HAS LOTS OF ENERGY', 's62b', '62.b FEELS CLOSE TO PEOPLE AT SCHOOL',
		's62c', '62.c SELDOM GETS SICK', 's62d', '62.d GETS BETTER QUICKLY', 's62e',
		'62.e FEELS PART OF SCHOOL', 's62f', '62.f IS WELL COORDINATED', 's62g',
		'62.g STUDENTS AT SCHOOL ARE PREJUDICED', 's62h', '62.h HAS GOOD QUALITIES', 's62i',
		'62.i HAPPY TO BE AT THIS SCHOOL', 's62j', '62.j PHYSICALLY FIT', 's62k',
		'62.k HAS A LOT TO BE PROUD OF', 's62l', '62.l TEACHERS TREAT STUDENTS FAIRLY',
		's62m', '62.m LIKES SELF', 's62n', '62.n DOING EVERYTHNG RIGHT', 's62o',
		'62.o FEELS SOCIALLY ACCEPTED', 's62p', '62.p FEELS LOVED AND WANTED', 's62q',
		'62.q FEELS SAFE IN NEIGHBORHOOD', 's62r', '62.r FEELS SAFE AT SCHOOL', 's63',
		'63. SWEAT FROM WORK, PLAY, EXERCISE', 's64', '64. BEEN IN FIGHTS LAST YEAR', 's65',
		'65. NEEDED TO GO TO DOCTOR BUT DID NOT', 's6a', '6.1 WHAT IS YOUR RACE (WHITE)', 's6b',
		'6.2 WHAT IS YOUR RACE? (BLACK)', 's6c', '6.3 WHAT IS YOUR RACE? (ASIAN)', 's6d',
		'6.4 WHAT IS YOUR RACE? (AMERICAN INDIAN)', 's6e', '6.5 WHAT IS YOUR RACE? (OTHER)',
		's8', '8. BORN IN THE UNITED STATES?')
	feature_map <- matrix(feature_map, ncol=2, nrow=length(feature_map)/2, byrow=T)
	
	map <- setNames(feature_map[, 2], feature_map[, 1])
	imp <- importance(rf)
	rownames(imp) <- map[rownames(imp)]
	imp
}

drop_columns <- function(df, to_drop){
	df[, !(names(df) %in% to_drop)]
}


remove_na <- function(data, vec){
    for(v in vec){
        tmp <- data[[v]]
        data <- data[!is.na(tmp),]
    }
    data
}