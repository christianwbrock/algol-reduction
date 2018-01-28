

argument_parser = ArgumentParser(parents=[time_parser('start-time'), time_parser('end-time')])
args = argument_parser.parse_args()

start = get_time_from_args(args, 'start-time')
end = get_time_from_args(args, 'end-time', required=True)

print(start.iso if start else " not defined")
print(end.iso if end else " not defined")
