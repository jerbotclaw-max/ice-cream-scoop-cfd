# First Bite: the impatient ice-cream utensil

## Engineering verdict

The Snow Peak titanium spork observation is credible, but titanium is not the
thermal trick. Titanium and 304 stainless are both poor conductors
(approximately 17 and 16 W/m·K). The Snow Peak wins the first instant because
its short, thin tines place the same hand force onto far less leading-edge area.

The best manufacturable passive utensil is therefore a hybrid:

- shallow 30–32 mm spork/spade bowl;
- three or four 3.5–4.5 mm cutting tines;
- 0.25–0.35 mm rounded micro-bevel;
- stiff titanium or hardened-stainless shell;
- broad thumb pad;
- 3 mm² copper spine from the grip to within 5 mm of the food-contact edge;
- copper fully clad so it never touches the food or lips.

That preserves the roughly 12× nominal edge-pressure advantage in the screening
model while moving about 0.7 W of hand heat toward the tip under the assumed
temperature gradient. It is the best balance of first-bite penetration, weight,
cleanability, cost, and durability.

## The exotic options

### CVD diamond: real property, wrong bottleneck

Thermal-grade CVD diamond reaches roughly 1,500–2,200 W/m·K. However,
conductance is `k × area / length`. A generous 20 µm film spread over a 10 mm
wide ribbon has only 0.20 mm² of axial cross-section. In the screen it transports
only about 0.2 W from hand to tip—better than bare titanium, worse than a cheap
3 mm² copper spine.

Diamond can still be useful as a local wear-resistant heat spreader on a premium
tip, but it is not the rational first design. A thick diamond insert would be
expensive, difficult to join, and brittle at the exact place a user pries.

### Heat pipe: fastest passive transport, but use the right fluid

A heat pipe can exhibit effective conductivity around 1,500–50,000 W/m·K.
A 3 mm pipe therefore makes axial conduction cease to be the bottleneck; the
hand/handle contact and the miniature pipe's power limit take over.

A standard copper/water electronics pipe is the wrong component. Water freezes
at the -18°C ice-cream end, and conventional water pipes do not operate well
until the assembly warms. A custom methanol low-temperature pipe can operate
around -20°C to 80°C (some designs quote -60°C to 80°C).

The result, **First Bite X**, could move about 3 W in an optimistic screening
case. That is only 75 mg of ice-cream-equivalent melt in five seconds, but it is
enough to create a lubricating melt film. The tines still do the initial fracture
work.

Practical problems:

- sealing a methanol heat pipe inside a dishwasher-safe mouth utensil;
- impact, bending, and repeated-cycle reliability;
- heat-pipe orientation and cold-start performance;
- higher manufacturing cost for a gain the user may get by running an ordinary
  thin-edged utensil under hot water for five seconds.

## Why the model does not claim a final product winner

`first_bite_model.py` is an engineering screen, not CFD or a product test. It
separates two first-order effects:

1. nominal leading-edge pressure at a fixed 5 N force;
2. steady axial hand-to-tip power, capped at 3 W by assumed hand contact.

It deliberately excludes:

- the utensil's initial room-temperature stored heat;
- contact resistance against a rough frozen surface;
- ice-cream overrun, fat, sugar, and exact freezer temperature;
- measured fracture strength;
- heat-pipe startup transients.

Run it with:

```bash
python3 first_bite_model.py
```

The next honest step is a force-gauge bench test at -18°C. Compare an ordinary
teaspoon, Snow Peak spork, thick aluminum ice-cream spoon, solid-copper spoon,
and a copper-spine prototype at 0, 10, and 30 seconds of hand warming.

## Published 15.0% temperature test

15.0% publishes a five-minute hand-warming comparison performed with the
Toyama Prefectural Industrial Technology Center. The standard solid-aluminum
15.0% spoon's handle rose 5.6°C and its tip rose 5.2°C; the conventional
stainless spoon's handle rose 2.5°C and its tip rose 0°C. This supports the
cross-section/geometry argument: a thick, uncoated solid body can outperform a
thin or poorly coupled utensil even when a competitor uses a nominally more
conductive metal.

It does **not** close the First Bite validation gap. The test starts near room
temperature, lasts five minutes, and reports temperatures rather than insertion
force against -18°C ice cream. The separate No.21 product uses solid copper;
its product page should not be presented as the specimen from the aluminum
performance test.

## Sources

- Snow Peak Titanium Spork:
  https://www.snowpeak.com/products/titanium-spork
- 15.0% thermal spoon third-party comparison:
  https://www.15percent.jp/about/performance/
- 15.0% solid-copper No.21 ex vanilla:
  https://www.15percent.jp/products/ice-cream-spoon/no21-ex-vanilla.html
- Coherent thermal-grade CVD diamond, 1,500 to over 2,200 W/m·K:
  https://www.coherent.com/content/dam/coherent/site/en/resources/datasheet/optics/thermal-cvd-diamond-ds.pdf
- Mersen heat-pipe effective conductivity, 1,500–50,000 W/m·K:
  https://www.mersen.com/en/products/cooling-solutions-services/understanding-heat-pipe-thermal-conductivity
- Advanced Cooling Technologies on water heat-pipe freezing/startup:
  https://www.1-act.com/thermal-solutions/passive/heat-pipes/heat-pipes-101/
- Celsia on methanol and acetone heat pipes:
  https://celsiainc.com/heat-sink-blog/methanol-heat-pipes-acetone-heat-pipes-machined-vapor-chambers/
- CoolingHouse methanol heat-pipe range:
  http://coolinghouse.com/technologies/methanol-heat-pipe/
