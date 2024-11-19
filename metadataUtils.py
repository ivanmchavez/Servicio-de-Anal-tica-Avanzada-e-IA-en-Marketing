import numpy as np
import re
import ast
import emoji

from sklearn.preprocessing import RobustScaler

IDEOGRAPHIC_SPACE = 0x3000  # Unicode point for ideographic space, used to determine if a character is Asian

def is_asian(char):
    """
    Check if a character is Asian based on its Unicode value.

    Args:
    - char: A single character.

    Returns:
    - Boolean indicating whether the character is Asian.
    """
    return ord(char) > IDEOGRAPHIC_SPACE

def filter_jchars(c):
    """
    Filter out Asian characters, replacing them with spaces.

    Args:
    - c: A character.

    Returns:
    - Space if the character is Asian; otherwise, the original character.
    """
    if is_asian(c):
        return ' '
    return c

def nonj_len(word):
    """
    Returns the number of non-Asian words in a given text.

    Args:
    - word: A string potentially containing Asian characters.

    Returns:
    - Number of non-Asian words.
    """
    # Convert Asian characters to spaces and split to count non-Asian words
    chars = [filter_jchars(c) for c in word]
    return len(''.join(chars).split())

def emoji_count(text):
    """
    Count the number of emojis in a given text.

    Args:
    - text: A string potentially containing emojis.

    Returns:
    - Number of emojis in the text.
    """
    return len([i for i in text if i in emoji.EMOJI_DATA])

def get_wordcount(text):
    """
    Get the word and character count for a given text, including Asian and emoji characters.

    Args:
    - text: The text of the segment.

    Returns:
    - A dictionary containing counts of various types of characters and words.
    """
    characters = len(text)
    chars_no_spaces = sum([not x.isspace() for x in text])
    asian_chars = sum([is_asian(x) for x in text])
    non_asian_words = nonj_len(text)
    emoji_chars = emoji_count(text)
    words = non_asian_words + asian_chars + emoji_chars

    return dict(characters=characters,
                chars_no_spaces=chars_no_spaces,
                asian_chars=asian_chars,
                non_asian_words=non_asian_words,
                emoji_chars=emoji_chars,
                words=words)

def dict2obj(dictionary):
    """
    Transform a dictionary into an object.

    Args:
    - dictionary: A dictionary.

    Returns:
    - An object with dictionary keys as attributes.
    """
    class Obj(object):
        def __init__(self, dictionary):
            self.__dict__.update(dictionary)
    return Obj(dictionary)

def get_wordcount_obj(text):
    """
    Get the word count as an object instead of a dictionary.

    Args:
    - text: The text of the segment.

    Returns:
    - An object containing word count information.
    """
    return dict2obj(get_wordcount(text))

def metadata(shorts_df):
    """
    Extract metadata features from a DataFrame containing video information.

    Args:
    - shorts_df: DataFrame containing video information such as tags, duration, title, etc.

    Returns:
    - A numpy array containing processed and scaled metadata features.
    """
    # Count number of tags for each video
    tags_cnt = []
    for tags in shorts_df['videoTags']:
        if tags == 'none':
            tags_cnt.append(0)
        else:
            tags_cnt.append(len(ast.literal_eval(tags)))

    # Convert video duration to seconds
    m4_sort = shorts_df['videoDuration'].tolist()
    m4_sort_sec = []
    for duration in m4_sort:
        try:
            m4_sort_sec.append(int(duration[2:4]))
        except:
            try:
                m4_sort_sec.append(int(duration[2:3]))
            except:
                continue

    # Calculate word count for video titles
    title_length = []
    for title in shorts_df['videoTitle']:
        cleaned_title = re.sub(r'[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+', "", title)
        word_count_obj = get_wordcount_obj(cleaned_title)
        title_length.append(word_count_obj.words)

    # Calculate word count for video descriptions
    descript_length = []
    for description in shorts_df['videoDescription']:
        try:
            cleaned_description = re.sub(r'[\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+', "", description)
        except TypeError:  # Handle cases where the description is NaN
            cleaned_description = ''
        word_count_obj = get_wordcount_obj(cleaned_description)
        descript_length.append(word_count_obj.words)

    # Scale metadata features using RobustScaler
    scaler = RobustScaler()
    user_metadata = scaler.fit_transform(np.array(shorts_df['subscriberCount']).reshape(-1, 1))
    vid_duration = scaler.fit_transform(np.array(m4_sort_sec).reshape(-1, 1))
    vid_tags = scaler.fit_transform(np.array(tags_cnt).reshape(-1, 1))
    title_length = scaler.fit_transform(np.array(title_length).reshape(-1, 1))
    descript_length = scaler.fit_transform(np.array(descript_length).reshape(-1, 1))
    totalViewCount = scaler.fit_transform(np.array(shorts_df['totalViewCount']).reshape(-1, 1))
    totalVideoCount = scaler.fit_transform(np.array(shorts_df['totalVideoCount']).reshape(-1, 1))
    averageViewCount = scaler.fit_transform((np.array(shorts_df['totalViewCount'] / shorts_df['totalVideoCount'])).reshape(-1, 1))

    # Combine and compute additional metadata features
    ver1 = (vid_duration + vid_tags) * (title_length + descript_length)
    ver2 = (user_metadata / vid_tags)
    ver3 = (user_metadata * vid_duration)
    ver4 = (totalVideoCount / totalViewCount)

    # Stack all metadata features into a single array
    metadata_arr = np.column_stack([user_metadata, totalViewCount, totalVideoCount, averageViewCount, 
                                    vid_duration, vid_tags, title_length, descript_length])

    print(metadata_arr.shape)  # Output the shape of the metadata array
    return metadata_arr
