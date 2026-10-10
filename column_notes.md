# Week 1 — Dataset Access & Key Column Notes (ds76)

**Access confirmed:** I downloaded 8 months of `CRMLSSold` files (Oct 2025 – May 2026) from `/raw/California` and loaded them in pandas. Together they have 166,725 rows and 78 columns. After filtering to `PropertyType = Residential` and `PropertySubType = SingleFamilyResidence`, there are **83,435 sales**. All rows have `MlsStatus = Closed`.

The definitions below follow the RESO / Trestle field definitions (see *Trestle Property MetaData.pdf*). "% missing" is measured on the single-family residence (SFR) subset.

## Target
| Column | Meaning | Notes |
|---|---|---|
| **ClosePrice** | Final price the property sold for. | **Target variable.** Median $890K. Has extreme outliers ($1.75 to $989.5M). |

## Identifiers
| Column | Meaning | Notes |
|---|---|---|
| ListingKey / ListingKeyNumeric | Unique system ID for the listing record. | Primary key. There are 60 duplicates across months, so dedupe on this. |
| ListingId | The MLS number agents see. | Not a feature. |

## Property type (filters)
| Column | Meaning | Notes |
|---|---|---|
| PropertyType | Broad class: Residential, ResidentialLease, Land, etc. | Keep only `Residential`. |
| PropertySubType | Sub-class: SingleFamilyResidence, Condominium, etc. | Keep only `SingleFamilyResidence`. |
| MlsStatus | Listing status in the MLS. | Always `Closed` in these files. |

## Size & structure (main predictors)
| Column | Meaning | Notes |
|---|---|---|
| **LivingArea** | Finished living space in sq ft. | Strongest structural predictor (ρ ≈ 0.52). |
| **BedroomsTotal** | Total number of bedrooms. | 42 rows have 0 bedrooms. |
| **BathroomsTotalInteger** | Total bathrooms (full + partial) as a whole number. | ρ ≈ 0.48. |
| LotSizeSquareFeet / LotSizeAcres / LotSizeArea | Lot size, in sq ft, in acres, and in the listing's own units. | Same information three times; keep the sq-ft version. Has extreme outliers. |
| YearBuilt | Year the home was built. | Use to engineer property age. |
| Stories / Levels | Number of floors / level description (One, Two, MultiSplit). | About 10% missing. |
| GarageSpaces, AttachedGarageYN, ParkingTotal | Garage capacity, whether the garage is attached, and total parking spaces. | ParkingTotal has some negative values. |
| MainLevelBedrooms | Bedrooms on the entry level. | 39% missing. |
| NewConstructionYN | Whether the home is newly built. | |

## Amenities (Y/N flags)
| Column | Meaning |
|---|---|
| PoolPrivateYN | Has a private pool. |
| ViewYN | Has a view (ocean, mountain, city, etc.). |
| FireplaceYN | Has a fireplace. |
| Flooring | Flooring materials (text list). 36% missing. |
| WaterfrontYN, BasementYN | More than 97% missing. Drop. |

## Location
| Column | Meaning | Notes |
|---|---|---|
| **Latitude / Longitude** | Property coordinates. | Needed for the Week 6 school-district spatial join. 42 rows fall outside California. |
| City, PostalCode, CountyOrParish, StateOrProvince | Address hierarchy. | County median price ranges from $330K (Lake) to $2.1M (San Mateo). |
| MLSAreaMajor | MLS-defined market area (e.g. "92126 - Mira Mesa"). | |
| UnparsedAddress, StreetNumberNumeric | Street address. | Not a feature. |
| SubdivisionName | Neighborhood or tract name. | 64% missing. |
| Elementary/MiddleOrJunior/HighSchool (+ District) | Assigned schools. | School names are more than 80% missing and the elementary/middle district columns are 100% empty, so use the Week 6 shapefile join instead. |

## Pricing & timing (be careful of leakage)
| Column | Meaning | Notes |
|---|---|---|
| ListPrice | Asking price at the time of sale. | ρ = 0.99 with ClosePrice. **Leakage risk**, so the team should decide whether it's allowed. |
| OriginalListPrice | First asking price. | Same leakage concern. |
| DaysOnMarket | Days between listing and going under contract. | Some values are negative. |
| ListingContractDate | Date the listing agreement started. | |
| PurchaseContractDate | Date the offer was accepted (went pending). | |
| CloseDate | Date the sale closed. | Use for the time-based train/test split. |
| ContractStatusChangeDate | Date of the last status change. | |

## HOA / tax
| Column | Meaning | Notes |
|---|---|---|
| AssociationFee, AssociationFeeFrequency | HOA dues and how often they're billed (Monthly, Annually). | Normalize to a monthly amount before using. |
| TaxAnnualAmount, TaxYear | Annual property tax. | 100% empty. Drop. |

## Agents & offices (not property characteristics)
`ListAgent*`, `CoListAgent*`, `BuyerAgent*`, `CoBuyerAgent*`, `ListOfficeName`, `BuyerOfficeName`, `CoListOfficeName`, `*AOR` (Association of Realtors). These describe who handled the sale, not the home. Exclude them from the model.

## Empty or near-empty (drop)
TaxYear, TaxAnnualAmount, BusinessType, CoveredSpaces, FireplacesTotal, AboveGradeFinishedArea, BelowGradeFinishedArea, ElementarySchoolDistrict, MiddleOrJuniorSchoolDistrict, WaterfrontYN, BasementYN, BuilderName, LotSizeDimensions, BuildingAreaTotal, CoBuyerAgentFirstName (all more than 90% missing).
