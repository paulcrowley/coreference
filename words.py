""" Word lists for coreference resolution """
import nltk


nounLabels = ['NN', 'NNS']

# Referential determiners
referentialDets = ['this', 'that', 'these', 'those', 'the']

# Names
names = nltk.corpus.names
male_names = names.words('male.txt')
female_names = names.words('female.txt')
person_names = male_names + female_names

# Pronouns
pronouns = ['he', 'she', 'him', 'her', 'it', 'his', 'hers', 'its', 'they', 'them', 'their']
male_pronouns = ["he", "him", "his"]
female_pronouns = ["she", "her", "hers"]
non_it_pronouns = ['he', 'she', 'him', 'her', 'his', 'hers', 'they', 'them', 'their']
poss_pronouns = ['his', 'her', 'hers', 'their', 'its']
reflexives = ['himself', 'herself', 'itself', 'themselves', 'theirselves']
plural_pronouns = ["they", "them"]
pro_forms = pronouns + poss_pronouns + reflexives

# Nouns
male_sg_nouns = ['spokesman','chairman','man','boy','boyfriend','brother','dad','dude', 'father', 'fiance', 'gentleman', 'god', 'grandfather', 'grandpa', 'grandson', 'groom', 'himself', 'husband', 'king', 'male', 'mr', 'nephew', 'nephews', 'priest', 'prince', 'son', 'uncle', 'waiter', 'master', 'widower', 'waiter']
prob_male_sg_nouns =  ['football player','knight','blacksmith','plumber','bricklayer','president','paratrooper','truck driver','firefighter','soldier','chauffeur','butcher','gangster','carpenter','janitor','skateboarder','body builder','captain','CEO','electrician','golfer','lieutenant','pilot','welder','locksmith','thief','contractor','engineer','rancher','police officer','surfer','farmer','groundskeeper','prime minister','president','programmer','ranger','undertaker','coach','surgeon','philosopher','funeral director','porter','accountant','chef','lawyer','actor','scientist','comedian']
total_male_nouns = male_sg_nouns + prob_male_sg_nouns

female_sg_nouns =  ['heroine','spokeswoman','chairwoman','woman','actress','aunt','bride','daughter','female','fiancee','girl','girlfriend','goddess','granddaughter','grandma','grandmother','lady','mom','mother','mrs','ms','niece','waitress','priestess','princess','queen','sister','waitress','mistress','widow','wife','woman']
prob_female_sg_nouns = ['designer', 'herbalist','masseuse','angel','nutritionist','sales sssistant','speech therapist','dental technician','personal assistant','interior decorator','yoga instructor','florist','fashion designer','gymnast','typist','figure skater','flight attendant','makeup artist','shopper','weaver','caregiver','librarian','fairy','model','wedding planner','nurse','dressmaker','elementary school teacher','knitter','fortune teller','hair dresser','wuilter','babysitter','ballet dancer','feminist','receptionist','housekeeper','house cleaner','secretary','seamstress','witch','beautician','nanny','midwife','cheerleader','maid']
total_female_nouns = female_sg_nouns + prob_female_sg_nouns

personNouns = total_male_nouns + total_female_nouns


