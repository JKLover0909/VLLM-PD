import React from 'react';
import { Loader2 } from "lucide-react";

export function EmployeeLogin({
  mode,
  modeText,
  t,
  verifyEmployeeCode,
  employeeCodeInput,
  updateEmployeeCodeInput,
  employeeCodeError,
  employeeVerifying,
}) {
  return (
    <div className="employee-gate">
      <form className="employee-card" onSubmit={verifyEmployeeCode}>
        <div className="employee-logo">
          <img src="/mkac-logo.png" alt="MKAC" />
        </div>
        <h1>{t("common.employeeAuth")}</h1>
        <p>{modeText(mode).authHint}</p>
        <label className="employee-field">
          <span>{t("common.employeeCode")}</span>
          <input
            value={employeeCodeInput}
            onChange={(event) => updateEmployeeCodeInput(event.target.value)}
            inputMode="numeric"
            autoComplete="off"
            maxLength={6}
            placeholder={t("common.employeeCode")}
            disabled={employeeVerifying}
            autoFocus
          />
        </label>
        {employeeCodeError && (
          <span className="employee-error" role="alert">
            {employeeCodeError}
          </span>
        )}
        <button
          className="employee-submit"
          type="submit"
          disabled={employeeVerifying}
        >
          {employeeVerifying && <Loader2 className="spin" size={16} />}
          {employeeVerifying ? t("common.verifying") : t("common.continue")}
        </button>
      </form>
    </div>
  );
}
