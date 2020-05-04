import spacy
import csv
import re
import json
from collections import defaultdict, Counter
from tqdm import tqdm

from bedstuy.consts import *

SKIP = {}
STOP_LIST = {"food", "groceries", "bottle", "frozen", "-PRON-", "bag", "can", "kind", "pack", "box", "supply", "kinds", "allergies"}
LEMMA_EXCEPTIONS = {"string"}
NLP = spacy.load('en_core_web_sm')

def normalize_span(span):
    return ' '.join([token.lemma_ if token.text not in LEMMA_EXCEPTIONS else token.text for token in span if not (token.is_punct or token.is_space or token.is_stop or token.text in STOP_LIST or token.lemma_ in STOP_LIST)]).strip("()")

def normalize_string(input_string):
    return input_string.lower().strip().strip("()")

def load_taxonomy():
    # load in initial taxonomy and aliases
    taxonomy_rows = []
    with open(TAXONOMY_PATH, encoding="utf8") as _csv_file:
        csv_reader = csv.reader(_csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            else:
                taxonomy_rows.append((normalize_string(row[0]), [normalize_string(item) for item in re.split(",|\.", row[2])]))

    canonical_to_aliases = {}
    for row in taxonomy_rows:
        original_aliases = [item for item in row[1] if item != '']
        normed_aliases = [alias for alias in [normalize_span(NLP(item)) for item in original_aliases] if alias != '']
        canonical = normalize_span(NLP(row[0]))
        if canonical.endswith('s'):
            normed_aliases.append(canonical[:-1])
        else:
            normed_aliases.append(canonical + 's')
        canonical_to_aliases[canonical] = {"original": original_aliases, "normed": normed_aliases}

    alias_to_canonical = {}
    for canonical, alias_dict in canonical_to_aliases.items():
        for alias in alias_dict['normed']:
            alias_to_canonical[alias] = canonical

    return canonical_to_aliases, alias_to_canonical

def load_grocery_lists():
    # load in grocery lists
    grocery_lists = []
    with open(GROCERIES_PATH, encoding="utf8") as _csv_file:
        csv_reader = csv.reader(_csv_file, delimiter=",")
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            else:
                grocery_lists.append(row[1])

    grocery_lists = [normalize_string(li) for li in grocery_lists if li != ""]
    return grocery_lists

def report(item_counter, still_unidentified_counter, canonical_to_aliases):
    print(f"{len(item_counter)} canonical items")
    print(f"{sum([sum(counter.values()) for item, counter in item_counter.items()])} occurrences of those canonical items")
    print(f"{len(still_unidentified_counter)} unidentified items")
    print(f"{sum(still_unidentified_counter.values())} occurrences of those unidentified items")

    output_json = []
    for item, counter in item_counter.items():
        item_output = {}
        item_output['name'] = item
        item_output['count'] = sum(counter.values())
        item_output['original_aliases'] = list(set(canonical_to_aliases[item]['original']))
        item_output['normed_and_added_aliases'] = sorted([(key, count) for key, count in counter.items() if count > 1 ], key=lambda x: x[1], reverse=True)
        output_json.append(item_output)
    output_json = sorted(output_json, key=lambda x: x['count'], reverse=True)
    with open(OUTPUT_PATH, 'w') as _json_file:
        json.dump(output_json, _json_file, indent=4)
    print(f"Output written to {OUTPUT_PATH}")

    unidentified_output_json = sorted([{'name': key, 'count': value} for key, value in still_unidentified_counter.items()], key=lambda x: x['count'], reverse=True)
    with open(UNIDENTIFIED_OUTPUT_PATH, 'w') as _json_file:
        json.dump(unidentified_output_json, _json_file, indent=4)
    print(f"Unidentified output written to {UNIDENTIFIED_OUTPUT_PATH}")

def exact_match(noun_chunked, item_counter, canonical_to_aliases, alias_to_canonical):
    unidentified_noun_chunks = []

    # simple cases on exact match to taxonomy or alias
    for li in noun_chunked:
        for noun_chunk in li:
            normed_chunk = normalize_span(noun_chunk)
            if normed_chunk in canonical_to_aliases:
                item_counter[normed_chunk][normed_chunk] += 1
            elif normed_chunk in alias_to_canonical:
                item_counter[alias_to_canonical[normed_chunk]][normed_chunk] += 1
            else:
                unidentified_noun_chunks.append(noun_chunk)

    # count up number of occurrences of unidentified items
    unidentified_counter = Counter()
    to_span_map = {}
    for unidentified_noun_chunk in unidentified_noun_chunks:
        normed_chunk = normalize_span(unidentified_noun_chunk)
        if normed_chunk == "":
            continue

        unidentified_counter[normed_chunk] += 1
        to_span_map[normed_chunk] = unidentified_noun_chunk

    return unidentified_counter, to_span_map

def token_match(unidentified_counter, to_span_map, canonical_to_aliases, alias_to_canonical, item_counter):
    # add frequently occurring unidentified items and try to normalize others
    still_unidentified_counter = Counter()
    for normed_chunk, count in unidentified_counter.items():
        item = to_span_map[normed_chunk]
        if count > 5:
            canonical_to_aliases[normed_chunk] = {"original": [], "normed": [normed_chunk]}
            item_counter[normed_chunk][normed_chunk] += count
        else:
            pos_tags = [token.pos_ for token in item]
            dep_tags = [token.dep_ for token in item]

            if 'ROOT' in dep_tags:
                root_token = item[dep_tags.index('ROOT')].lemma_.strip("()")

                if root_token in canonical_to_aliases:
                    item_counter[root_token][normed_chunk] += 1
                    canonical_to_aliases[root_token]["normed"].append(normed_chunk)
                elif root_token in alias_to_canonical:
                    item_counter[alias_to_canonical[root_token]][normed_chunk] += 1
                    canonical_to_aliases[alias_to_canonical[root_token]]["normed"].append(normed_chunk)
            else:
                found = False
                all_n_grams = get_all_n_grams(item)
                for n_gram in all_n_grams:
                    key = normalize_span(n_gram)
                    if key in canonical_to_aliases:
                        item_counter[key][normed_chunk] += 1
                        canonical_to_aliases[key]["normed"].append(normed_chunk)
                        found = True
                        break
                    elif key in alias_to_canonical:
                        item_counter[alias_to_canonical[key]][normed_chunk] += 1
                        canonical_to_aliases[alias_to_canonical[key]]["normed"].append(normed_chunk)
                        found = True
                        break
                if not found and item.text not in SKIP:
                    still_unidentified_counter[item.text] += 1
                # found = False
                # for token in item:
                #     key = token.lemma_.strip("()")
                #     if key in canonical_to_aliases:
                #         item_counter[key][normed_chunk] += 1
                #         canonical_to_aliases[key]["normed"].append(normed_chunk)
                #         found = True
                #         break
                #     elif key in alias_to_canonical:
                #         item_counter[alias_to_canonical[key]][normed_chunk] += 1
                #         canonical_to_aliases[alias_to_canonical[key]]["normed"].append(normed_chunk)
                #         found = True
                #         break
                # if not found and item.text not in SKIP:
                #     still_unidentified_counter[item.text] += 1
    return still_unidentified_counter

def get_n_grams(span, n):
    n_grams = []
    for i in range(len(span)):
        if (i+n) > len(span):
            continue
        n_grams.append(span[i:i+n])
    return n_grams

def get_all_n_grams(span):
    all_n_grams = []
    for n in range(1, len(span)+1):
        all_n_grams += get_n_grams(span, n)
    return all_n_grams[::-1]

def main():
    canonical_to_aliases, alias_to_canonical = load_taxonomy()
    grocery_lists = load_grocery_lists()

    noun_chunked = [list(NLP(li).noun_chunks) for li in tqdm(grocery_lists)]

    item_counter = defaultdict(Counter)
    unidentified_counter, to_span_map = exact_match(noun_chunked, item_counter, canonical_to_aliases, alias_to_canonical)
    still_unidentified_counter = token_match(unidentified_counter, to_span_map, canonical_to_aliases, alias_to_canonical, item_counter)


    report(item_counter, still_unidentified_counter, canonical_to_aliases)

    print("Done.")


if __name__ == "__main__":
    main()
