# experiment

similarity_cnn: 
step 2500 DEV loss=0.165217 acc=0.765 cm[[5050 1095][ 757  967]] P=0.469 R=0.561 F1=0.510829 *
# pos_weight=3.5
step 10800 DEV loss=0.306637 acc=0.615 cm[[3376 2769][ 260 1464]] P=0.346 R=0.849 F1=0.491523 * 

# sequence_length=20 filter_sizes=2,3,4,5 num_filters=100
step 4800 DEV loss=0.219696 acc=0.796 cm[[5406  739][ 863  861]] P=0.538 R=0.499 F1=0.518050 *

# sequence_length=50 filter_sizes=2,3,4,5 num_filters=100
step 5900 DEV loss=0.218425 acc=0.797 cm[[5411  734][ 862  862]] P=0.540 R=0.500 F1=0.519277 *
# char_level=True sequence_length=50 filter_sizes=2,3,4,5 num_filters=100
step 4400 DEV loss=0.213174 acc=0.798 cm[[5411  734][ 852  872]] P=0.543 R=0.506 F1=0.523724 *
# char_level=False sequence_length=50 filter_sizes=2,3,4,5 num_filters=100， pos_weight=2
step 9300 DEV loss=0.222963 acc=0.738 cm[[4582 1563][ 496 1228]] P=0.440 R=0.712 F1=0.543965 *
# pos_weight=3.5
step 8300 DEV loss=0.297538 acc=0.665 cm[[3826 2319][ 318 1406]] P=0.377 R=0.816 F1=0.516058 *


similarity_rnn: 0.51(euc) 0.535(cos)  0.494(ma)
step 4600 DEV loss=0.159456 acc=0.771 cm[[4992 1153][ 648 1076]] P=0.483 R=0.624 F1=0.544397 *
step 7600 DEV loss=0.151597 acc=0.791 cm[[5418  727][ 914  810]] P=0.527 R=0.470 F1=0.496780 *  embed_dropout 0.6 with input_dropout
step 7900 DEV loss=0.143206 acc=0.800 cm[[5781  364][1211  513]] P=0.585 R=0.298 F1=0.394464    embed_dropout 0.7 with varitional dropout(improve loss but not f1)
step 9400 DEV loss=0.28018 acc=0.739 cm[[4526 1619][ 434 1290]] P=0.443 R=0.748 F1=0.556875 * pos_weight=3.5
step 3700 DEV loss=0.218844 acc=0.785 cm[[5205  940][ 754  970]] P=0.508 R=0.563 F1=0.533847 * pos_weight=2

# hidden_units=50 num_layers=2 dropout=0.8 pos_weight=2
step 5500 DEV loss=0.231672 acc=0.709 cm[[4321 1824][ 465 1259]] P=0.408 R=0.730 F1=0.523819 * 
step 7500 DEV loss=0.233937 acc=0.716 cm[[4396 1749][ 484 1240]] P=0.415 R=0.719 F1=0.526204 *

# hidden_units=50 num_layers=2 dropout=1.0 pos_weight=2
step 7200 DEV loss=0.220919 acc=0.761 cm[[4985 1160][ 717 1007]] P=0.465 R=0.584 F1=0.517605 *
step 8600 DEV loss=0.221685 acc=0.768 cm[[5032 1113][ 709 1015]] P=0.477 R=0.589 F1=0.526999 *

# hidden_units=100 num_layers=2 dropout=0.7 pos_weight=2
step 5600 DEV loss=0.217153 acc=0.771 cm[[2488  572][ 330  544]] P=0.487 R=0.622 F1=0.546734 *



# dyanamic=True, hidden_units=100 num_layers=2 dropout=0.7 pos_weight=2, fast and overfitting
step 10600 DEV loss=0.21977 acc=0.772 cm[[2505  555][ 341  533]] P=0.490 R=0.610 F1=0.543323 *
# dyanamic=True, hidden_units=100 num_layers=2 dropout=0.5 pos_weight=2, weight_decay=0.5
step 6000 DEV loss=0.225551 acc=0.757 cm[[2464  596][ 360  514]] P=0.463 R=0.588 F1=0.518145 *
# dyanamic=True, hidden_units=100 num_layers=2 dropout=0.5 pos_weight=2 weight_decay=0.5 gru
step 4100 DEV loss=0.228839 acc=0.753 cm[[2423  637][ 335  539]] P=0.458 R=0.617 F1=0.525854 *
# dyanamic=True, hidden_units=100 num_layers=2 dropout=0.5 pos_weight=2 weight_decay=0.5 gru lr=1e-2
step 2400 DEV loss=0.221263 acc=0.765 cm[[2462  598][ 327  547]] P=0.478 R=0.626 F1=0.541852 *
# euclidean
step 1000 DEV loss=0.237447 acc=0.758 cm[[2482  578][ 374  500]] P=0.464 R=0.572 F1=0.512295 *


classification_cnn: 
step 7300 DEV loss=0.493472 acc=0.782 cm[[5325  820][ 898  826]] P=0.502 R=0.479 F1=0.490208 *
step 6800 DEV loss=0.457115 acc=0.796 cm[[5560  585][1024  700]] P=0.545 R=0.406 F1=0.465271 *

classification_rnn:
failed rnn output half zero ?