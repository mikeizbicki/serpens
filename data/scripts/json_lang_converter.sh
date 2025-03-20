#!/bin/sh

filepath="$1"
lang="$2"

#llm_command="llm -m claude-3-5-sonnet-20241022"
llm_command="llm -m groq-llama-3.3-70b"

$llm_command <<EOF
Translate the list elements in the following JSON into "$lang".
Use the proper language-specific alphabet, diacritics, and punctuation.
As much as possible, use simple grammar and common words.
Be consistent in grammar/vocabulary choice throughout the translations.
Keep the keys of the dictionary the same and only translate the values.
The output should be only the translated JSON with no explanation and should not be in a markdown code block.
The output JSON should contain a new field at the end "translation_notes" whose value is a markdown formatted explanation of any interesting aspects of the translation;
this explanation should be written in English.

$(cat "$filepath")
EOF
