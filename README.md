## lokigi - Generate, rank, analyse and visualise the best candidate solutions for facility location problems

lokigi means 'to locate' or 'to place' in the Esperanto language.

(or it's the backronym '**L**ocation **O**ptimisation: **K**-best solution **I**nspection, **G**eneration & **I**nsights' - whichever floats your boat)

lokigi exists to make the process of providing decision support for healthcare problems with a geographical component easier.

A range of fantastic libraries exist for geographic optimization (e.g. [spopt](https://pysal.org/spopt/)), but many use linear programming to optimize the solution, meaning you emerge with a single optimal solution. For healthcare contexts, this often isn't ideal - decision makers need a range of near-optimal solutions to balance against real-world constraints.

Building on [work from Dr Tom Monks](https://github.com/health-data-science-OR/healthcare-logistics/tree/master/optimisation) and [previous rounds of the Health Service Modelling Associates programme](https://hsma.co.uk/hsma_content/modules/current_module_details/3_geographic_modelling_visualisation.html), lokigi is designed to make it easier to beginner programmers to tackle location optimization problems for the benefit of their organisations.

Lokigi is planned to expand to boundary optimisation problems in the future.

# Getting started

Head to the [documentation](https://hsma-tools.github.io/lokigi/lokigi_docs/) to find out how to use the package.

For now, you would need to clone this repository to make use of it.

PyPi and Conda support coming very soon.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://hsma.co.uk"><img src="https://avatars.githubusercontent.com/u/29951987?v=4?s=100" width="100px;" alt="Sammi Rosser"/><br /><sub><b>Sammi Rosser</b></sub></a><br /><a href="https://github.com/hsma-tools/lokigi/commits?author=Bergam0t" title="Code">💻</a> <a href="https://github.com/hsma-tools/lokigi/commits?author=Bergam0t" title="Documentation">📖</a> <a href="https://github.com/hsma-tools/lokigi/commits?author=Bergam0t" title="Tests">⚠️</a> <a href="https://github.com/hsma-tools/lokigi/issues?q=author%3ABergam0t" title="Bug reports">🐛</a> <a href="#content-Bergam0t" title="Content">🖋</a> <a href="#design-Bergam0t" title="Design">🎨</a> <a href="#example-Bergam0t" title="Examples">💡</a> <a href="#ideas-Bergam0t" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-Bergam0t" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-Bergam0t" title="Maintenance">🚧</a> <a href="#projectManagement-Bergam0t" title="Project Management">📆</a> <a href="#research-Bergam0t" title="Research">🔬</a> <a href="#tutorial-Bergam0t" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://experts.exeter.ac.uk/19244-thomas-monks"><img src="https://avatars.githubusercontent.com/u/881493?v=4?s=100" width="100px;" alt="Tom Monks"/><br /><sub><b>Tom Monks</b></sub></a><br /><a href="https://github.com/hsma-tools/lokigi/commits?author=TomMonks" title="Code">💻</a> <a href="#ideas-TomMonks" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-TomMonks" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://sites.google.com/nihr.ac.uk/hsma"><img src="https://avatars.githubusercontent.com/u/43324262?v=4?s=100" width="100px;" alt="Dr Daniel Chalk"/><br /><sub><b>Dr Daniel Chalk</b></sub></a><br /><a href="#mentoring-hsma-chief-elf" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/amyheather"><img src="https://avatars.githubusercontent.com/u/92166537?v=4?s=100" width="100px;" alt="Amy Heather"/><br /><sub><b>Amy Heather</b></sub></a><br /><a href="#infra-amyheather" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


## Note

This package builds on work from the metapy project from Dr Tom Monks.

Metapy can be found [here](https://github.com/health-data-science-OR/healthcare-logistics/tree/master/optimisation/metapy).

Metapy is release under the MIT licence. The licence is reproduced below in line with the terms of the licence.

> MIT License

> Copyright (c) 2020 health-data-science-OR

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.

> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Modified metapy code is noted within the source code.
