import pandas as pd
import numpy as np
import re

# The full text block of statistics you provided
STATS_TEXT = """
J
J candidate for log transform
[ 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5
12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5
24.5 25.5 26.5 27.5 28.5 29.5 30.5 31.5 32.5 33.5 34.5 35.5
36.5 37.5 38.5 39.5 40.5 41.5 42.5 43.5 44.5 45.5 46.5 47.5
48.5 49.5 50.5 51.5 52.5 53.5 54.5 55.5 56.5 57.5 58.5 59.5
60.5 61.5 62.5 63.5 64.5 65.5 66.5 67.5 68.5 69.5 70.5 71.5
72.5 73.5 74.5 75.5 76.5 77.5 78.5 79.5 80.5 81.5 82.5 83.5
84.5 0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22.
23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34.
35. 36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46.
47. 48. 49. 50. 51. 52. 53. 54. 55. 56. 57. 58.
59. 60. 61. 62. 63. 64. 65. 66. 67. 68. 69. 70.
71. 72. 73. 74. 75. 76. 77. 78. 79. 80. 81. 82.
83. 84. 85. 86. 87. 88. 89. 90. 91. 92. 93. 94.
95. 96. 97. 98. 99. 100. 101. 102. 103. 104. 105. 106.
107. 108. 109. 110. 111. 112. 113. 114. 115. 116. 117. 118.
119. 120. 121. 122. 123. 124. 125. 126. 127. 128. 129. 130.
131. 132. 133. 134. 135. 136. 137. 138. 139. 140. 141. 142.
143. 144. 145. 146. 147. 148. 149. 150. 151. 152. 153. 154.
155. 156. 157. 158. 159. 160. 161. 162. 163. 164. 165. 166.
167. 168. 170. ]
31.030324039507647
25.740586501097205
170.0
170000000000.0
Jpp
Jpp candidate for log transform
[ 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5
12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5
24.5 25.5 26.5 27.5 28.5 29.5 30.5 31.5 32.5 33.5 34.5 35.5
36.5 37.5 38.5 39.5 40.5 41.5 42.5 43.5 44.5 45.5 46.5 47.5
48.5 49.5 50.5 51.5 52.5 53.5 54.5 55.5 56.5 57.5 58.5 59.5
60.5 61.5 62.5 63.5 64.5 65.5 66.5 67.5 68.5 69.5 70.5 71.5
7T2.5 73.5 74.5 75.5 76.5 77.5 78.5 79.5 80.5 81.5 82.5 83.5
1. 0. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11.
12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23.
24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.
36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47.
48. 49. 50. 51. 52. 53. 54. 55. 56. 57. 58. 59.
60. 61. 62. 63. 64. 65. 66. 67. 68. 69. 70. 71.
72. 73. 74. 75. 76. 77. 78. 79. 80. 81. 82. 83.
84. 85. 86. 87. 88. 89. 90. 91. 92. 93. 94. 95.
96. 97. 98. 99. 100. 101. 102. 103. 104. 105. 106. 107.
108. 109. 110. 111. 112. 113. 114. 115. 116. 117. 118. 119.
120. 121. 122. 123. 124. 125. 126. 127. 128. 129. 130. 131.
132. 133. 134. 135. 136. 137. 138. 139. 140. 141. 142. 143.
144. 145. 146. 147. 148. 149. 150. 151. 152. 153. 154. 155.
156. 157. 158. 159. 160. 161. 162. 163. 164. 165. 166. 168. ]
31.121477483627366
25.829730393326646
168.0
168000000000.0
Kc_aprox
Kc_aprox candidate for log transform
[ 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5
12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5
24.5 25.5 26.5 27.5 28.5 29.5 30.5 31.5 32.5 33.5 34.5 35.5
36.5 37.5 38.5 39.5 40.5 41.5 42.5 43.5 44.5 45.5 46.5 47.5
48.5 49.5 50.5 51.5 52.5 53.5 54.5 55.5 56.5 57.5 58.5 59.5
60.5 61.5 62.5 63.5 64.5 65.5 66.5 67.5 68.5 69.5 70.5 71.5
72.5 73.5 74.5 75.5 76.5 77.5 78.5 79.5 80.5 81.5 82.5 83.5
84.5 0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22.
23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34.
35. 36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46.
47. 48. 49. 50. 51. 52. 53. 54. 55. 56. 57. 58.
59. 60. 61. 62. 63. 64. 65. 66. 67. 68. 69. 70.
71. 72. 73. 74. 75. 76. 77. 78. 79. 80. 81. 82.
83. 84. 85. 86. 87. 88. 89. 90. 91. 92. 93. 94.
95. 96. 97. 98. 99. 100. 101. 102. 103. 104. 105. 106.
107. 108. 109. 110. 111. 112. 113. 114. 115. 116. 117. 118.
119. 120. 121. 122. ]
20.430989611109926
18.042034319231043
122.0
122000000000.0
Kcpp_aprox
Kcpp_aprox candidate for log transform
[ 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5
12.5 13.5 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5
24.5 25.5 26.5 27.5 28.5 29.5 30.5 31.5 32.5 33.5 34.5 35.5
36.5 37.5 38.5 39.5 40.5 41.5 42.5 43.5 44.5 45.5 46.5 47.5
48.5 49.5 50.5 51.5 52.5 53.5 54.5 55.5 56.5 57.5 58.5 59.5
60.5 61.5 62.5 63.5 64.5 65.5 66.5 67.5 68.5 69.5 70.5 71.5
72.5 73.5 74.5 75.5 76.5 77.5 78.5 79.5 80.5 81.5 82.5 83.5
1. 0. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11.
12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23.
24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.
36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47.
48. 49. 50. 51. 52. 53. 54. 55. 56. 57. 58. 59.
60. 61. 62. 63. 64. 65. 66. 67. 68. 69. 70. 71.
72. 73. 74. 75. 76. 77. 78. 79. 80. 81. 82. 83.
84. 85. 86. 87. 88. 89. 90. 91. 92. 93. 94. 95.
96. 97. 98. 99. 100. 101. 102. 103. 104. 105. 106. 107.
108. 109. 110. 111. 112. 113. 114. 115. 116. 117. 118. 119.
120. 121. -1. ]
20.717846705207243
18.051486656871763
122.0
121000000000.0
active_weight
[ 50.96825 35.976793 34.0329 16.0425 28.0101 18.0153
30.026 20.00689 2.01588 31.9988 17.0305 60.075
93.9529 15.0234751 28.0134 44.071 17.0073 127.91241
79.926277 44.0095 53.97324 28.0532 80.063 30.0061
32.0419 141.939 64.064 146.0554 47.0134 95.92568
26.0373 ]
59.788027995437
34.47520192655294
144.03951999999998
72.45242772387245
active_dipole
active_dipole candidate for log transform
[1.2974 1.109 1.847 0. 0.11011 1.857
2.331 1.826526 1.4719 0.715 1.81 1.957
1.6552 0.45 0.828 1.51 0.159 1.67229184
1.62 1.6331 1.813 1.38 ]
1.263191884902191
0.6694628639357803
2.331
2331000000.0
active_m
[2. 6. 4.]
2.372037459735051
0.7888157465898005
4.0
3.0
active_polar
[ 3.02 2.515 2.54 2.448 1.953 1.501
2.77 0.8 0.787 1.562 2.103 5.09
5.61 1.298 1.71 5.39684463 1.136 5.453
3.616 2.507 3.438 4.188 19.029 1.698
3.21 7.325 3.882 4.49 2.247 2.215
3.487 ]
5.8205867533489775
6.003808857448476
18.242
24.179161372299873
active_B0a
active_B0a candidate for log transform
[1.00000000e+05 5.18200000e+00 5.24120000e+00 2.78806313e+01
9.40552614e+00 9.94664516e+00 5.24631000e+00 9.57789000e+00
2.04636200e+01 4.82800000e+00 3.46000000e-01 4.25240851e+00
5.17340000e+00 2.02735420e+00 9.10700000e-02 3.09855000e+00
2.04700700e+01]
8875.775677831696
28418.308738311312
99999.90893
1098056.4401010212
active_B0b
active_B0b candidate for log transform
[ 0.62048896 10.4401993 0.85179 5.2412 1.92252895 14.5217696
1.29536237 20.5597234 60.853 1.43767 9.94664516 0.20285674
0.32192 9.57789 1.67195 0.817084 18.5519677 6.42637
8.34923939 0.39021 0.50424 1.0012 0.346 1.69614207
0.82323552 0.25021 0.3441739 0.09107 0.41778 0.3528
1.176646 ]
2.8520887587241077
6.123393390700508
60.76193
668.2002854946744
active_B0c
active_B0c candidate for log transform
[ 0.62048896 10.4401993 0.85179 5.2412 1.92252895 9.27770838
1.13425949 20.5597234 60.853 1.43767 6.22750356 0.20285674
0.32192 4.74202 1.67195 0.817084 18.5519677 6.42637
8.34923939 0.39021 0.4912 0.8282 0.173 1.69614207
0.79287185 0.25021 0.2935265 0.09107 0.36749 0.34634
1.176646 ]
2.107070074795529
4.929916649084222
60.76193
668.2002854946744
perturber_weight
[ 50.96825 35.976793 16.0425 28.0134 31.9988
28.97 146.0554 2.01588 44.0095 60.075
2.33375552 18.0153 44.071 28.0101 53.97324
79.926277 20.00689 17.0305 80.063 32.0419
34.0329 76.141 30.0061 93.9529 95.92568 ]
46.25508550610492
22.13809598906112
144.03951999999998
72.45242772387245
perturber_dipole
perturber_dipole candidate for log transform
[1.2974 1.109 0. 0.715 1.857 1.957
0.11011 1.51 0.828 1.826526 1.4719 1.67229184
1.847 0.159 1.81 1.38 ]
0.4433315252887345
0.7069811328375902
1.957
1957000000.0
perturber_m
[2. 6. 4. 2.64]
3.3837438402628157
0.9246518402080735
4.0
3.0
perturber_polar
[ 3.02 2.515 2.448 1.71 1.562 1.67892
4.49 0.787 2.507 5.09 0.69436 1.501
5.39684463 1.953 3.438 3.616 0.8 2.103
19.029 3.21 2.54 8.749 1.698 5.61
2.215 ]
5.021020470523712
6.215324269211801
18.33464
27.405092459243043
perturber_B0a
perturber_B0a candidate for log transform
[1.00000000e+05 5.24120000e+00 9.10700000e-02 2.78806313e+01
2.04636200e+01 9.94664516e+00 3.46000000e-01 4.25240851e+00
5.18200000e+00 5.24631000e+00 2.04700700e+01]
56933.96303895003
49513.23049591897
99999.90893
1098056.4401010212
perturber_B0b
perturber_B0b candidate for log transform
[6.20488960e-01 1.04401993e+01 5.24120000e+00 1.67195000e+00
1.43767000e+00 1.62275120e+00 9.10700000e-02 6.08530000e+01
3.90210000e-01 2.02856740e-01 1.60511165e+04 1.45217696e+01
8.17084000e-01 1.92252895e+00 5.04240000e-01 8.34923939e+00
2.05597234e+01 9.94664516e+00 3.46000000e-01 8.23235520e-01
8.51790000e-01 1.09100000e-01 1.69614207e+00 3.21920000e-01
3.52800000e-01]
5.137575970618213
222.4964823795018
16051.02543
176250.31843636764
perturber_B0c
perturber_B0c candidate for log transform
[ 0.62048896 10.4401993 5.2412 1.67195 1.43767 1.6227512
0.09107 60.853 0.39021 0.20285674 51.3761602 9.27770838
0.817084 1.92252895 0.4912 8.34923939 20.5597234 6.22750356
0.173 0.79287185 0.85179 0.1091 1.69614207 0.32192
0.34634 ]
2.0096452846624135
7.838751617215097
60.76193
668.2002854946744
m
[ 4. 8. 10. 6. 8.64 12. ]
5.755781299997867
1.3826269910282432
8.0
3.0
d_act_per
[384.2 333.9 376.5 372. 363. 370.2 444.4 287.25 342.75 413.8
324.7 338. 368.2 370. 361. 332. 413. 332.02 334.5 331.95
395. 386. 371.5 363.6 352.5 386.7 353.2 346. 355. 285.5
326.5 358. 376. 312.5 380.2 311.45 347.85 330. 307.75 354.95
380. 320. 335.3 339.5 314.8 377. 351.5 375. 347.15 348.95
339.95 470. 303. 356.95 351. 373. 416.5 344.7 455.2 448.
457. 370.6 317. 301.9 366.7 368.5 429.4 438.4 436.6 442.9
423.35 289. 399.5 330.2 300. 369.5 360.5 340.2 330.75 329.
323.5 353. 347. ]
364.4451005823752
60.510376091066966
184.5
1.6462346760070052
is_self
is_self candidate for log transform
[1. 0.]
0.46031102672952623
0.4984276126143417
1.0
999999999.9999999
broadness_jeanna
[0.06049039 0.05438049 0.05920685 0.0560216 0.05211999 0.05514621
0.06320568 0.12677301 0.04787982 0.06365723 0.11915375 0.03636004
0.0557398 0.05675657 0.05231977 0.16984114 0.06438352 0.10394785
0.11531308 0.038427 0.05211274 0.04731758 0.04248722 0.05830627
0.0550182 0.05248072 0.04412344 0.04127771 0.04467116 0.05618405
0.12061224 0.05648214 0.07815196 0.1504867 0.04614754 0.14524111
0.04071358 0.03201732 0.04110102 0.06567769 0.06328546 0.04077859
0.03679119 0.13040189 0.06481883 0.04752233 0.05235715 0.04375836
0.09969431 0.0466762 0.04760546 0.04357088 0.04815153 0.13865993
0.05196981 0.05275517 0.06273792 0.18249688 0.06977337 0.07399752
0.04726332 0.0655435 0.06094009 0.0669931 0.04172132 0.05367046
0.13933063 0.05274073 0.05374875 0.0496352 0.05467328 0.05346877
0.05793135 0.05141102 0.17210159 0.04930319 0.04571761 0.06561478
0.03586361 0.02688417 0.06509057 0.06048193 0.04135837 0.06834818
0.05594409 0.038572 0.04673175 0.04520578 0.04158464]
0.04869702437904038
0.017964871123889375
0.15561271439317187
6.7882659485782515
active_rms_quadrupole
active_rms_quadrupole candidate for log transform
[0.78954903 2.71717568 0.98994949 0. 2.00771188 2.09633013
0.24859606 1.6541585 0.36769553 0.2192031 1.64048773 0.41295036
2.51258831 0.46244783 0.98570685 0.07071068 1.3576325 3.70806796
2.83596962 3.02500281 1.70683411 2.23345025 4.41046071 0.76412215
0.84357839 3.90888629 3.65367851 2.92049556 1.95366544 4.47975717]
2.217336343470166
1.273824000638161
4.47975717347864
4479757173.47864
perturber_rms_quadrupole
perturber_rms_quadrupole candidate for log transform
[0.78954903 2.71717568 0. 0.98570685 0.2192031 0.83259471
0.36769553 3.02500281 0.41295036 0.30886424 2.09633013 0.07071068
2.00771188 1.70683411 2.83596962 1.6541585 1.64048773 4.41046071
0.84357839 0.98994949 2.43810418 0.76412215 2.51258831 1.95366544]
1.8982145144023421
1.3677063991630964
4.410460709117208
4410460709.117208
"""

# --- NEW ---
# Define the exact feature order your model expects.
# I've corrected the typos from your input list.
EXPECTED_FEATURE_ORDER = [
    'J',
    'Jpp',
    'Kc_aprox',
    'Kcpp_aprox',
    'active_B0a',
    'active_B0b',
    'active_B0c',
    'active_dipole',
    'active_m',
    'active_polar',
    'active_rms_quadrupole',
    'active_weight',
    'broadness_jeanna',
    'd_act_per',
    'is_self',
    'm',
    'perturber_B0a',
    'perturber_B0b',
    'perturber_B0c',
    'perturber_dipole',
    'perturber_m',
    'perturber_polar',
    'perturber_rms_quadrupole',
    'perturber_weight'
]


# -----------


def create_synthetic_dataset(text_data, n_samples=1000):
    """
    Parses a block of text containing feature statistics and generates a
    synthetic DataFrame.

    Args:
        text_data (str): The multi-line string containing the stats.
        n_samples (int): The number of samples (rows) to generate.

    Returns:
        pd.DataFrame: A synthetic DataFrame.
    """

    print("Parsing feature statistics...")

    # This regex captures one full feature block
    # It uses re.MULTILINE (for ^) and re.DOTALL (for .* to span newlines)
    pattern = re.compile(
        r"^(?P<name>[a-zA-Z0-9_]+)\n" +  # Group 'name': Feature name
        # --- FIX ---
        # Changed '.*' to '[^\n]*' to prevent greedy matching across lines
        r"(?:[^\n]*candidate for log transform\n)?" +  # Optional log line
        # -----------
        r"^(?P<uniques>\[.*?\])\n" +  # Group 'uniques': The [ ... ] list
        r"^(?P<mean>[\d\.\-e\+]+)\n" +  # Group 'mean': The mean value
        r"^(?P<std>[\d\.\-e\+]+)\n" +  # Group 'std': The std dev value
        r"^(?P<range>[\d\.\-e\+]+)\n" +  # Group 'range': The range (max - min)
        r"^(?P<ratio>[\d\.\-e\+]+)",  # Group 'ratio': The ratio
        re.MULTILINE | re.DOTALL
    )

    parsed_stats = {}

    for match in pattern.finditer(text_data):
        try:
            stats = match.groupdict()
            feature_name = stats['name']

            # Extract main stats
            mean = float(stats['mean'])
            std = float(stats['std'])

            # Parse the uniques list to find min, max, and type
            uniques_str = stats['uniques']
            # Find all floating point numbers, including scientific notation
            numbers_str = re.findall(r"[\d\.\-e\+]+", uniques_str)

            if not numbers_str:
                print(f"Warning: No numbers found in 'uniques' for {feature_name}. Skipping.")
                continue

            numbers = [float(f) for f in numbers_str]
            min_val = min(numbers)
            max_val = max(numbers)

            # Heuristic for discreteness:
            # If '...' is NOT in the string, the list is complete.
            # Treat it as discrete and sample from it.
            # Note: This logic is imperfect but matches the parser's intent
            if '...' not in uniques_str:
                is_discrete = True
                unique_vals = np.array(numbers)
            else:
                is_discrete = False
                unique_vals = None

            parsed_stats[feature_name] = {
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'is_discrete': is_discrete,
                'unique_vals': unique_vals
            }

        except Exception as e:
            print(f"Error parsing block for {stats.get('name', 'UNKNOWN')}: {e}")

    print(f"Successfully parsed {len(parsed_stats)} features.")

    # --- 2. Build the DataFrame ---
    print(f"Generating synthetic DataFrame with {n_samples} samples...")

    dataset = {}
    for feature, s in parsed_stats.items():
        if s['is_discrete']:
            # For discrete features, sample from their unique values
            print(f"  -> Generating '{feature}' (discrete) by sampling from {s['unique_vals']}")
            dataset[feature] = np.random.choice(s['unique_vals'], size=n_samples, replace=True)
        else:
            # For continuous features, sample from Normal dist and clip
            print(
                f"  -> Generating '{feature}' (continuous) from N(mean={s['mean']:.2f}, std={s['std']:.2f}) and clipping to [{s['min']}, {s['max']}]")
            # Ensure std dev is non-negative
            std_dev = max(0, s['std'])
            data_col = np.random.normal(s['mean'], std_dev, size=n_samples)
            dataset[feature] = np.clip(data_col, s['min'], s['max'])

    # --- MODIFIED ---
    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    # Check if all expected columns were parsed and generated
    generated_cols = set(df.columns)
    expected_cols_set = set(EXPECTED_FEATURE_ORDER)

    missing_in_data = list(expected_cols_set - generated_cols)
    if missing_in_data:
        print(
            f"Warning: The following expected features were not found in the parsed stats and will be missing: {missing_in_data}")

    extra_in_data = list(generated_cols - expected_cols_set)
    if extra_in_data:
        print(
            f"Warning: The following features were parsed but are not in the expected list and will be dropped: {extra_in_data}")

    # Re-order the DataFrame to match the expected feature order
    # This selects only the columns in EXPECTED_FEATURE_ORDER that are also in df.columns
    final_ordered_cols = [col for col in EXPECTED_FEATURE_ORDER if col in generated_cols]

    return df[final_ordered_cols]
    # --- END MODIFICATION ---



def create_synthetic_dataset_low_J(text_data, n_samples=1000):
    """
    Parses a block of text containing feature statistics and generates a
    synthetic DataFrame. J, Jpp, Kc_aprox, and Kcpp_aprox are forced to be low discrete values.
    """

    print("Parsing feature statistics...")

    pattern = re.compile(
        r"^(?P<name>[a-zA-Z0-9_]+)\n"
        r"(?:[^\n]*candidate for log transform\n)?"
        r"^(?P<uniques>\[.*?\])\n"
        r"^(?P<mean>[\d\.\-e\+]+)\n"
        r"^(?P<std>[\d\.\-e\+]+)\n"
        r"^(?P<range>[\d\.\-e\+]+)\n"
        r"^(?P<ratio>[\d\.\-e\+]+)",
        re.MULTILINE | re.DOTALL
    )

    parsed_stats = {}
    for match in pattern.finditer(text_data):
        try:
            stats = match.groupdict()
            feature_name = stats["name"]
            mean = float(stats["mean"])
            std = float(stats["std"])
            uniques_str = stats["uniques"]

            numbers_str = re.findall(r"[\d\.\-e\+]+", uniques_str)
            if not numbers_str:
                print(f"Warning: No numbers found in uniques for {feature_name}. Skipping.")
                continue

            numbers = [float(f) for f in numbers_str]
            min_val, max_val = min(numbers), max(numbers)

            if "..." not in uniques_str:
                is_discrete = True
                unique_vals = np.array(numbers)
            else:
                is_discrete = False
                unique_vals = None

            parsed_stats[feature_name] = {
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "is_discrete": is_discrete,
                "unique_vals": unique_vals,
            }
        except Exception as e:
            print(f"Error parsing block for {stats.get('name','UNKNOWN')}: {e}")

    print(f"Successfully parsed {len(parsed_stats)} features.")
    print(f"Generating synthetic DataFrame with {n_samples} samples...")

    # Hard override values for rotational quantum numbers
    forced_small_qnums = ["J", "Jpp", "Kc_aprox", "Kcpp_aprox"]
    small_values = np.array([1, 2, 3, 4, 5])

    dataset = {}
    for feature, s in parsed_stats.items():

        # If feature is in list, force discrete sampling 1–5
        if feature in forced_small_qnums:
            print(f"  -> Forcing '{feature}' to discrete {small_values}")
            dataset[feature] = np.random.choice(small_values, size=n_samples, replace=True)
            continue

        # Otherwise keep original behaviour
        if s["is_discrete"]:
            print(f"  -> Generating '{feature}' (discrete) from parsed unique values")
            dataset[feature] = np.random.choice(s["unique_vals"], size=n_samples, replace=True)
        else:
            std_dev = max(0, s["std"])
            print(f"  -> Generating '{feature}' (continuous) from N(mean={s['mean']:.2f}, std={std_dev:.2f})")
            col = np.random.normal(s["mean"], std_dev, size=n_samples)
            dataset[feature] = np.clip(col, s["min"], s["max"])

    df = pd.DataFrame(dataset)

    generated_cols = set(df.columns)
    expected_cols_set = set(EXPECTED_FEATURE_ORDER)
    final_ordered_cols = [c for c in EXPECTED_FEATURE_ORDER if c in generated_cols]

    return df[final_ordered_cols]

if __name__ == "__main__":
    # 1. Generate the dataset
    # You can change the number of samples here
    synthetic_data = create_synthetic_dataset(STATS_TEXT, n_samples=5000)

    # 2. Display the results
    print("\n--- Synthetic Dataset Head ---")
    print(synthetic_data.head())

    print("\n--- Synthetic Dataset Description (Stats) ---")
    # Use .describe() to show the stats of our generated data
    # This is a good sanity check!
    print(synthetic_data.describe())

    # 3. Save to CSV (Optional)
    output_filename = "synthetic_feature_dataset.csv"
    synthetic_data.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved dataset to {output_filename}")