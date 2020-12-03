import argparse
import json

from gym_gridverse.envs.yaml.schemas import env_schema

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indent', type=int, default=4)
    args = parser.parse_args()

    json_schema = env_schema().json_schema('TO-BE-REMOVED')
    # remove mandatory $id field ('TO-BE-REMOVED')
    del json_schema['$id']

    print(json.dumps(json_schema, indent=args.indent))
