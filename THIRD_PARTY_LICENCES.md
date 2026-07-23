This package builds on work from the metapy project from Dr Tom Monks.

Metapy can be found [here](https://github.com/health-data-science-OR/healthcare-logistics/tree/master/optimisation/metapy).

Metapy is release under the MIT licence. The licence is reproduced below in line with the terms of the licence.

> MIT License
>
> Copyright (c) 2020 health-data-science-OR
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Modified metapy code is noted within the source code.

---

`tests/test_two_step_floating_catchment.py` cross-validates `two_step_floating_catchment()`'s gravity-weighted (`distance_decay={"method": "power", ...}`) accessibility calculation against the small hospital-accessibility example from the [pysal/access](https://github.com/pysal/access) project (`access/tests/test_hospital_example.py`).

pysal/access is released under the BSD 3-Clause licence. The licence is reproduced below in line with the terms of the licence.

> BSD 3-Clause License
>
> Copyright 2018 pysal-access developers
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**Extent of reuse.** No pysal/access source code is included in lokigi -- `test_two_step_floating_catchment.py` contains only new lokigi test code that builds a `SiteProblem` via lokigi's own API. What is reused from pysal/access is:

- Their small example **dataset**: three locations with specific population, doctor-count and travel-cost values, taken directly from `test_hospital_example.py`'s four cost-matrix scenarios.
- The resulting **expected accessibility values**, computed by independently running pysal/access's own `simple_2sfca` reference function (also defined in `test_hospital_example.py`, not copied into lokigi) over each scenario -- not copied from any lokigi output.

This lets lokigi's generalised weight-matrix engine be checked against a real, independently maintained implementation's numbers, rather than only lokigi's own hand-derived arithmetic.
