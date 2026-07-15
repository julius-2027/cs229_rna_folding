find fpt_results/ -maxdepth 1 -type f | while read f; do
    [[ $(wc -l < "$f") -lt 1000 ]] && rm "$f"
done
