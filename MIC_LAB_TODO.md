
## After PR #624 lab measurement — strategy refresh

* **Chat path is now grace-bounded.** `mesh_chat` p50 floor on the public
  mesh is ~6s = the `first_answer_grace` config. To make chat feel
  faster, lower the grace (1.5-2s), trade off: more likely to take a
  first-answer that might be lower-quality than what a longer wait
  would give. **Cheap experiment in the lab.**
* **Reducer path is where option A matters.** On the public mesh
  post-fix, reducer fires 0% of chat turns (grace catches everything).
  On lab tool turns it fires 36% of the time. Streaming would help
  tool turns feel responsive.
* **Tool turns currently buffer reducer output.** Real streaming would
  show the reducer's tokens as it produces them. ~150 LoC.

