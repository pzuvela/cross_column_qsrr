echo "Running  QSRR test framework"

TEST_PATH=$(pwd)
cd ..
cd ..
ROOT_PATH=$(echo $(pwd)/qsrr)
cd $TEST_PATH

export COVERAGE_PROCESS_START=$(echo $TEST_PATH.coveragerc)

PYTHON_PATH="$(echo $TEST_PATH):$(echo $ROOT_PATH)"

if [ -z $PYTHONPATH ]; then
    export PYTHONPATH=$PYTHON_PATH
else
    export PYTHONPATH=$PYTHONPATH:$PYTHON_PATH
fi

echo "$(echo $ROOT_PATH)"
echo "$(echo $TEST_PATH)"
echo "$(echo $PYTHONPATH)"

coverage run --source "$(echo $ROOT_PATH)" -m pytest -v "$(echo $TEST_PATH)"
coverage report -m --omit='*tests*,*notebooks*,*data*'