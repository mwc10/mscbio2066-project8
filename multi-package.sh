Help()
{
    echo "Batch compress multiple models in tar.gz files"
    echo
    echo $0 "[Options] -s [dir]"
    echo
    echo "Options:"
    echo "-h | --help     show this help message"
    echo "-d | --dry-run  print commands to run"
    echo "-s | --sync     directory containing configs/ and models/ subdirs"
    echo
}

DRY_RUN=""
while getopts ":hds:" option; do
  case $option in
    d | dry-run)
      DRY_RUN=echo
      ;;
    s | sync)
      SYNC=${OPTARG}
      ;;
    h | help | *)
      Help
      exit;;
  esac
done

configs=$SYNC/configs
data=$SYNC/data
models=$SYNC/models
for model in "$models"/*/; do
  name=$(basename $model)
  config=$configs/$name.json
  $DRY_RUN python3 package-model.py -m $model -c $config -d $data --tar
  echo Bundled $name
done
