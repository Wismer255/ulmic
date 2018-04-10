# -*- coding: utf-8 -*-
import click
from ulmic.environment import UlmicEnvironment as ue

@click.group()
def main(args=None):
    """Console script for ulmic"""
    click.echo("Replace this message by putting your code into "
               "ulmic.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")

@click.command()
def hello():
    click.echo('Hello World!')

@click.command()
def env():
    click.echo(ue.get_home_dir())
    click.echo(ue.get_data_dir())
    click.echo(ue.get_test_dir())
    click.echo(ue.get_log_dir())

main.add_command(env)

if __name__ == "__main__":
    main()
    hello()
