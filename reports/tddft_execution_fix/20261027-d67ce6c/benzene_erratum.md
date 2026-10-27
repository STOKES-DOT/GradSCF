# Historical benzene correction

The first solve DID return: response_solve=318.490191238001s;
end_to_end=352.6166828069836s. Repeat1 then finished SCF in24.063490s.
The controller enforced600s over the ENTIRE four-repeat subprocess, not each
kernel invocation. It sent SIGTERM during repeat1 response; returncode=-15,
process wall600.151501s. Thus the prior report's first-TDA-timeout statement
was false. Original JSONL and process.json remain read-only.

Finite first energies and a completed stage do not imply a completed repeated
process or certify per-root residuals. Historical residual_per_root was not
recorded; scientific_pass is unknown and accepted_for_comparison is false.
New runs separately budget single solves/processes and certify the original
operator after clean timing/memory capture.
